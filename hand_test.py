import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
from collections import deque

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ----------------------------
# Basic helpers
# ----------------------------
def distance(p1, p2):
    """Straight-line distance between two 2D points."""
    return np.linalg.norm(p1 - p2)


def finger_extended(tip, knuckle):
    """
    Non-thumb finger is 'extended' if fingertip is above knuckle in the image.
    (Works best when palm faces camera.)
    """
    return tip[1] < knuckle[1]


def finger_folded(tip, pip, mcp, wrist, hand_scale):
    """
    Folded finger heuristic combining vertical and distance checks.
    More tolerant to slight hand rotation than y-only tests.
    """
    tip_below_pip = tip[1] > pip[1] - 0.02 * hand_scale
    tip_near_mcp = distance(tip, mcp) < 0.92 * hand_scale
    tip_near_wrist = distance(tip, wrist) < 1.70 * hand_scale
    return tip_below_pip or (tip_near_mcp and tip_near_wrist)


def stable_index_draw_pose(points):
    """
    Draw pose: index finger clearly extended while middle/ring/pinky are folded.
    Made extra forgiving to maintain continuous drawing during hand micro-movements.
    """
    wrist = points[0]
    hand_scale = distance(wrist, points[9]) + 1e-6

    index_tip, index_pip, index_mcp = points[8], points[6], points[5]
    middle_tip, middle_pip, middle_mcp = points[12], points[10], points[9]
    ring_tip, ring_pip, ring_mcp = points[16], points[14], points[13]
    pinky_tip, pinky_pip, pinky_mcp = points[20], points[18], points[17]

    # Very relaxed thresholds for continuous writing (was 0.02, now 0.01)
    index_up = index_tip[1] < index_pip[1] - 0.01 * hand_scale
    index_long = distance(index_tip, index_mcp) > 0.75 * hand_scale  # Relaxed from 0.82

    middle_folded = finger_folded(middle_tip, middle_pip, middle_mcp, wrist, hand_scale)
    ring_folded = finger_folded(ring_tip, ring_pip, ring_mcp, wrist, hand_scale)
    pinky_folded = finger_folded(pinky_tip, pinky_pip, pinky_mcp, wrist, hand_scale)

    return index_up and index_long and middle_folded and ring_folded and pinky_folded


def compute_hand_length(points):
    """
    Normalized hand length (stable vs camera distance).
    wrist->middle_tip divided by wrist->index_knuckle.
    """
    wrist = points[0]
    middle_tip = points[12]
    index_knuckle = points[5]
    return distance(wrist, middle_tip) / (distance(wrist, index_knuckle) + 1e-6)


def detect_gesture(points):
    """
    Rule-based gesture detection.
    Returns: "FIST", "OPEN", "POINT", "PEACE", "THUMBS_UP", "THUMBS_DOWN", "PINCH", "UNKNOWN"
    """
    wrist = points[0]
    thumb_tip = points[4]
    index_tip, index_pip = points[8], points[6]
    middle_tip, middle_pip = points[12], points[10]
    ring_tip, ring_pip = points[16], points[14]
    pinky_tip, pinky_pip = points[20], points[18]

    index = finger_extended(index_tip, index_pip)
    middle = finger_extended(middle_tip, middle_pip)
    ring = finger_extended(ring_tip, ring_pip)
    pinky = finger_extended(pinky_tip, pinky_pip)

    hand_scale = distance(wrist, points[9]) + 1e-6

    # Thumb up/down relative to wrist (scaled, not fixed pixels)
    thumb_up = thumb_tip[1] < wrist[1] - 0.25 * hand_scale
    thumb_down = thumb_tip[1] > wrist[1] + 0.25 * hand_scale
    thumb_extended = thumb_up or thumb_down

    # Robust-ish fist detection
    curled_fingers = sum([not index, not middle, not ring, not pinky])
    avg_tip_dist = np.mean([
        distance(index_tip, wrist),
        distance(middle_tip, wrist),
        distance(ring_tip, wrist),
        distance(pinky_tip, wrist)
    ]) / hand_scale
    thumb_close = distance(thumb_tip, wrist) < hand_scale * 1.2

    if curled_fingers >= 3 and avg_tip_dist < 1.5 and thumb_close:
        return "FIST"

    if all([thumb_extended, index, middle, ring, pinky]):
        return "OPEN"

    # Index-up pose: draw mode trigger (thumb can be either open or closed)
    if index and not middle and not ring and not pinky:
        return "POINT"

    if index and middle and not ring and not pinky:
        return "PEACE"

    if thumb_up and not any([index, middle, ring, pinky]):
        return "THUMBS_UP"

    if thumb_down and not any([index, middle, ring, pinky]):
        return "THUMBS_DOWN"

    if distance(thumb_tip, index_tip) < hand_scale * 0.4:
        return "PINCH"

    return "UNKNOWN"


# ----------------------------
# HUD drawing pieces
# ----------------------------
def draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=2, L=18):
    cv2.line(img, (x1, y1), (x1 + L, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + L), color, thickness)
    cv2.line(img, (x2, y1), (x2 - L, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + L), color, thickness)
    cv2.line(img, (x1, y2), (x1 + L, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - L), color, thickness)
    cv2.line(img, (x2, y2), (x2 - L, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - L), color, thickness)


def draw_waveform(img, origin, t, color):
    ox, oy = origin
    w = 110
    amp = 10
    pts = []
    for i in range(w):
        y = oy + int(math.sin((i * 0.14) + t * 3.2) * amp * 0.6 + math.sin((i * 0.05) + t * 2.0) * amp * 0.4)
        pts.append((ox + i, y))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2)


def draw_mini_bar_graph(img, origin, states, color):
    ox, oy = origin
    bar_w, gap = 6, 4
    max_h = 28
    for i, on in enumerate(states):
        h = max_h if on else 10
        x = ox + i * (bar_w + gap)
        cv2.rectangle(img, (x, oy - h), (x + bar_w, oy), color, -1)
        cv2.rectangle(img, (x, oy - h), (x + bar_w, oy), (255, 255, 255), 1)


def gesture_mode(gesture):
    return {
        "OPEN":        "ERASE MODE",
        "FIST":        "COMBAT MODE",
        "PEACE":       "ANALYZE MODE",
        "POINT":       "DRAW MODE",       # ← updated label
        "THUMBS_UP":   "CONFIRM",
        "THUMBS_DOWN": "DENY",
        "PINCH":       "PINCH CTRL"
    }.get(gesture, "IDLE")


def mode_color(gesture):
    if gesture == "FIST":        return (0, 0, 255)
    if gesture == "OPEN":        return (0, 120, 255)
    if gesture == "PEACE":       return (255, 255, 0)
    if gesture in ("THUMBS_UP", "THUMBS_DOWN"): return (255, 200, 50)
    if gesture == "POINT":       return (180, 100, 255)   # purple pen color
    return (0, 255, 255)


# ----------------------------
# Particles
# ----------------------------
def spawn_particles(particles, center, n=18):
    cx, cy = center
    for _ in range(n):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2.0, 7.0)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        life = random.randint(18, 35)
        size = random.randint(2, 4)
        particles.append([cx, cy, vx, vy, life, size])


def update_and_draw_particles(frame, particles):
    alive = []
    for p in particles:
        x, y, vx, vy, life, size = p
        x += vx
        y += vy
        vy += 0.06
        life -= 1
        if life > 0:
            cv2.circle(frame, (int(x), int(y)), max(1, int(size * (life / 35))), (255, 255, 255), -1)
            alive.append([x, y, vx, vy, life, size])
    particles[:] = alive


# ----------------------------
# Main HUD
# ----------------------------
def draw_hud_v3(frame, center, bbox, hand_length, gesture, finger_states, t):
    x, y = center
    x1, y1, x2, y2 = bbox

    accent = mode_color(gesture)
    mode = gesture_mode(gesture)

    hud = frame.copy()

    draw_corner_brackets(hud, x1, y1, x2, y2, accent, thickness=2, L=20)

    scan = int((math.sin(t * 2.8) * 0.5 + 0.5) * (y2 - y1))
    scan_y = y1 + scan
    cv2.line(hud, (x1, scan_y), (x2, scan_y), (255, 255, 255), 1)

    radius = int(58 + hand_length * 15)
    rot = (t * 140) % 360

    for a in range(0, 360, 18):
        aa = math.radians(a + rot)
        r1 = radius + 5
        r2 = radius + (18 if a % 36 == 0 else 11)
        p1 = (x + int(math.cos(aa) * r1), y + int(math.sin(aa) * r1))
        p2 = (x + int(math.cos(aa) * r2), y + int(math.sin(aa) * r2))
        cv2.line(hud, p1, p2, accent, 2)

    cv2.ellipse(hud, (x, y), (radius, radius), int(rot), 0, 255, accent, 2)
    cv2.circle(hud, (x, y), radius - 16, (0, 255, 255), 1)

    prog = int(min(hand_length * 95, 300))
    cv2.ellipse(hud, (x, y), (radius - 22, radius - 22), 0, 0, prog, (0, 255, 0), 3)

    cv2.line(hud, (x - 16, y), (x + 16, y), (0, 255, 255), 1)
    cv2.line(hud, (x, y - 16), (x, y + 16), (0, 255, 255), 1)

    panel_w, panel_h = 220, 105
    px, py = x2 + 14, y1
    if px + panel_w > frame.shape[1]:
        px = x1 - panel_w - 14

    cv2.rectangle(hud, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.rectangle(hud, (px, py), (px + panel_w, py + panel_h), accent, 2)

    cv2.putText(hud, f"{mode}", (px + 10, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(hud, f"LEN: {hand_length:.2f}", (px + 10, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    s1 = 100 + int(60 * math.sin(t * 1.6))
    s2 = 120 + int(40 * math.cos(t * 1.1))
    cv2.putText(hud, f"S1:{s1:03d}  S2:{s2:03d}", (px + 10, py + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    draw_mini_bar_graph(hud, (px + 12, py + panel_h + 35), finger_states, accent)
    draw_waveform(hud, (x - 55, y2 + 34), t, (0, 255, 255))

    cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)


# ----------------------------
# Drawing colors per hand
# Each hand cycles through a palette independently
# ----------------------------
DRAW_PALETTES = {
    "Right": [
        (180, 100, 255),   # purple
        (0,   200, 255),   # yellow-ish
        (0,   255, 128),   # green-cyan
        (50,  50,  255),   # red-orange
    ],
    "Left": [
        (255, 180,  50),   # blue-gold
        (255,  50, 200),   # magenta
        (100, 255,  80),   # lime
        (0,   200, 255),   # orange
    ],
}

draw_color_idx = {"Right": 0, "Left": 0}


def next_draw_color(label):
    """Advance and return the next pen color for this hand."""
    palette = DRAW_PALETTES[label]
    draw_color_idx[label] = (draw_color_idx[label] + 1) % len(palette)
    return palette[draw_color_idx[label]]


def current_draw_color(label):
    return DRAW_PALETTES[label][draw_color_idx[label]]


# ----------------------------
# Draw cursors for pen/eraser tools
# ----------------------------
def draw_pen_cursor(frame, tip, color, brush_size):
    cv2.circle(frame, tip, brush_size + 4, (255, 255, 255), 1)
    cv2.circle(frame, tip, brush_size, color, -1)


def draw_eraser_cursor(frame, tip, eraser_size):
    cv2.circle(frame, tip, eraser_size + 3, (255, 255, 255), 1)
    cv2.circle(frame, tip, eraser_size, (20, 20, 20), 2)


def square_view(frame, size):
    """
    Letterbox into a square canvas, then scale to requested display size.
    """
    h, w = frame.shape[:2]
    side = max(h, w)
    pad = np.zeros((side, side, 3), dtype=frame.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = frame

    if side != size:
        interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        pad = cv2.resize(pad, (size, size), interpolation=interp)
    return pad


# ----------------------------
# Main loop
# ----------------------------
trail_frames = deque(maxlen=6)
particles    = []

prev_gestures   = {"Left": "UNKNOWN", "Right": "UNKNOWN"}
gesture_buffers = {
    "Left":  deque(maxlen=5),
    "Right": deque(maxlen=5),
}

# Persistent pen tip position per hand (for smooth line segments)
prev_tip = {"Left": None, "Right": None}
prev_eraser_tip = {"Left": None, "Right": None}

# Position smoothing buffers (reduces jitter for smoother lines)
tip_position_buffer = {
    "Left": deque(maxlen=3),   # Smooth over last 3 positions
    "Right": deque(maxlen=3),
}

# Track how many consecutive draw-pose frames we've seen (lift-pen logic)
point_streak = {"Left": 0, "Right": 0}
prev_draw_pose = {"Left": False, "Right": False}
draw_hit_count = {"Left": 0, "Right": 0}
draw_miss_count = {"Left": 0, "Right": 0}

# Brush thickness (can be adjusted with PEACE gesture hold — future feature)
BRUSH_SIZE = 4
ERASER_SIZE = 18
DRAW_ACTIVATE_FRAMES = 0   # Instant activation (was 1) - start drawing immediately
DRAW_RELEASE_GRACE = 25    # Very forgiving (was 8) - allows ~0.8sec of gesture wobble

# ---- Canvas ----
# Initialized once we know the frame size.
canvas     = None
canvas_bgr = None   # color canvas (draw colored strokes on it)

# ---- Clear all canvas: require holding THUMBS_DOWN for N consecutive frames ----
CLEAR_GESTURE = "THUMBS_DOWN"
CLEAR_HOLD_FRAMES = 18
clear_hold = {"Left": 0, "Right": 0}

WINDOW_NAME = "Iron Man Hand HUD v3  |  ESC to quit"
WINDOW_SIZE = 960
cap         = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)
start_time = time.time()

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        t     = time.time() - start_time

        # ---- Lazy canvas init ----
        if canvas is None:
            canvas     = np.zeros((H, W), dtype=np.uint8)      # unused now; kept for mask option
            canvas_bgr = np.zeros((H, W, 3), dtype=np.uint8)   # persistent color drawing layer

        # ---- TRAIL ----
        if trail_frames:
            trail = frame.copy()
            for k, old in enumerate(trail_frames):
                a = 0.06 + 0.02 * k
                cv2.addWeighted(old, a, trail, 1 - a, 0, trail)
            frame = trail

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ---- Composite canvas onto frame BEFORE HUD/particles ----
        # cv2.add does per-pixel saturating addition: black (0,0,0) pixels on
        # canvas_bgr add nothing; colored stroke pixels light up on top of frame.
        cv2.add(frame, canvas_bgr, frame)

        # ---- Particles ----
        update_and_draw_particles(frame, particles)

        seen_labels = set()
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                seen_labels.add(label)

                points = np.array(
                    [[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark],
                    dtype=np.float32
                )

                raw_gesture = detect_gesture(points)
                hand_length = compute_hand_length(points)

                buf = gesture_buffers[label]
                buf.append(raw_gesture)
                gesture = max(set(buf), key=buf.count) if buf else raw_gesture

                idx_state   = finger_extended(points[8],  points[6])
                mid_state   = finger_extended(points[12], points[10])
                ring_state  = finger_extended(points[16], points[14])
                pinky_state = finger_extended(points[20], points[18])
                finger_states = [idx_state, mid_state, ring_state, pinky_state]
                draw_candidate = stable_index_draw_pose(points) or gesture == "POINT"
                open_pose = idx_state and mid_state and ring_state and pinky_state

                if open_pose:
                    draw_hit_count[label] = 0
                    draw_miss_count[label] = 0
                    draw_pose = False
                else:
                    if draw_candidate:
                        draw_hit_count[label] += 1
                        draw_miss_count[label] = 0
                    else:
                        draw_hit_count[label] = 0
                        if prev_draw_pose[label]:
                            draw_miss_count[label] += 1
                        else:
                            draw_miss_count[label] = 0

                    draw_pose = draw_candidate and draw_hit_count[label] >= DRAW_ACTIVATE_FRAMES
                    if prev_draw_pose[label] and draw_miss_count[label] <= DRAW_RELEASE_GRACE:
                        draw_pose = True

                wx, wy = int(points[0][0]), int(points[0][1])

                x1 = int(np.min(points[:, 0]) - 10)
                y1 = int(np.min(points[:, 1]) - 10)
                x2 = int(np.max(points[:, 0]) + 10)
                y2 = int(np.max(points[:, 1]) + 10)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ---- Particle burst on entering FIST ----
                if prev_gestures.get(label, "UNKNOWN") != "FIST" and gesture == "FIST":
                    spawn_particles(particles, (wx, wy), n=22)

                # ---- DRAWING LOGIC ----
                index_tip = (int(points[8][0]), int(points[8][1]))
                
                # Boundary check: Keep index tip within screen bounds with margin
                MARGIN = 20
                index_tip_x = max(MARGIN, min(W - MARGIN, index_tip[0]))
                index_tip_y = max(MARGIN, min(H - MARGIN, index_tip[1]))
                index_tip_bounded = (index_tip_x, index_tip_y)
                
                # Add to smoothing buffer
                tip_position_buffer[label].append(index_tip_bounded)
                
                # Use smoothed position (average of last 3 positions)
                if len(tip_position_buffer[label]) >= 2:
                    smoothed_x = int(np.mean([p[0] for p in tip_position_buffer[label]]))
                    smoothed_y = int(np.mean([p[1] for p in tip_position_buffer[label]]))
                    index_tip_smoothed = (smoothed_x, smoothed_y)
                else:
                    index_tip_smoothed = index_tip_bounded
                
                # Check if finger is near screen edge (to avoid drawing at boundaries)
                near_edge = (index_tip[0] < MARGIN or index_tip[0] > W - MARGIN or 
                            index_tip[1] < MARGIN or index_tip[1] > H - MARGIN)

                if draw_pose:
                    clear_hold[label] = 0
                    prev_eraser_tip[label] = None

                    # Advance color on the first frame after entering draw pose
                    if not prev_draw_pose[label]:
                        next_draw_color(label)
                        prev_tip[label] = None  # lift pen when re-entering mode
                        tip_position_buffer[label].clear()  # Clear smoothing buffer on mode change

                    pen_color = current_draw_color(label)
                    point_streak[label] += 1

                    # Draw line from previous tip to current tip for smooth strokes
                    # Only draw if:
                    # 1. Previous tip exists
                    # 2. Not starting fresh (point_streak allows immediate drawing now)
                    # 3. Distance is reasonable (avoid jumps)
                    # 4. Not near screen edge
                    if (prev_tip[label] is not None and 
                        not near_edge):
                        
                        # Calculate distance to avoid drawing extremely long lines
                        tip_distance = distance(np.array(prev_tip[label]), np.array(index_tip_smoothed))
                        MAX_DRAW_DISTANCE = 150  # pixels - adjust if needed
                        
                        if tip_distance < MAX_DRAW_DISTANCE:
                            cv2.line(canvas_bgr, prev_tip[label], index_tip_smoothed,
                                     pen_color, BRUSH_SIZE * 2, lineType=cv2.LINE_AA)
                            # Also draw on frame directly so it's visible this frame
                            cv2.line(frame, prev_tip[label], index_tip_smoothed,
                                     pen_color, BRUSH_SIZE * 2, lineType=cv2.LINE_AA)
                        else:
                            # Distance too large - lift pen to avoid long line
                            prev_tip[label] = None
                            tip_position_buffer[label].clear()

                    # Only update prev_tip if not near edge
                    if not near_edge:
                        prev_tip[label] = index_tip_smoothed
                    else:
                        prev_tip[label] = None  # lift pen at edges
                        tip_position_buffer[label].clear()
                        
                    prev_draw_pose[label] = True

                    # Draw glowing cursor (use smoothed position)
                    if not near_edge:
                        draw_pen_cursor(frame, index_tip_smoothed, pen_color, BRUSH_SIZE)

                elif open_pose:
                    clear_hold[label] = 0
                    prev_draw_pose[label] = False
                    prev_tip[label] = None
                    point_streak[label] = 0

                    eraser_tip = (int(points[9][0]), int(points[9][1]))

                    # Erase continuously while open palm is visible
                    if prev_eraser_tip[label] is not None:
                        cv2.line(canvas_bgr, prev_eraser_tip[label], eraser_tip,
                                 (0, 0, 0), ERASER_SIZE * 2, lineType=cv2.LINE_AA)
                    cv2.circle(canvas_bgr, eraser_tip, ERASER_SIZE, (0, 0, 0), -1, lineType=cv2.LINE_AA)

                    prev_eraser_tip[label] = eraser_tip
                    draw_eraser_cursor(frame, eraser_tip, ERASER_SIZE)

                else:
                    # Not POINT → lift the pen so next stroke starts fresh
                    # But only if we've been out of draw pose for more than grace period
                    if prev_draw_pose[label] and draw_miss_count[label] > DRAW_RELEASE_GRACE:
                        prev_draw_pose[label] = False
                        prev_tip[label] = None
                        point_streak[label] = 0
                    elif not prev_draw_pose[label]:
                        prev_tip[label] = None
                        point_streak[label] = 0
                        
                    prev_eraser_tip[label] = None

                    # ---- Hold clear gesture to clear canvas ----
                    if gesture == CLEAR_GESTURE:
                        clear_hold[label] += 1
                        if clear_hold[label] >= CLEAR_HOLD_FRAMES:
                            canvas_bgr[:] = 0
                            clear_hold[label] = 0
                            spawn_particles(particles, (wx, wy), n=40)
                    else:
                        clear_hold[label] = 0

                prev_gestures[label] = gesture

                # ---- HUD ----
                draw_hud_v3(frame, (wx, wy), (x1, y1, x2, y2),
                            hand_length, gesture, finger_states, t)

                # ---- Clear-hold progress indicator ----
                if gesture == CLEAR_GESTURE and clear_hold[label] > 0:
                    pct = clear_hold[label] / CLEAR_HOLD_FRAMES
                    bar_w = int(pct * (x2 - x1))
                    cv2.rectangle(frame, (x1, y2 + 6), (x1 + bar_w, y2 + 14), (0, 80, 255), -1)
                    cv2.rectangle(frame, (x1, y2 + 6), (x2,         y2 + 14), (0, 80, 255), 1)
                    cv2.putText(frame, "HOLD THUMBS DOWN TO CLEAR",
                                (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1)

                # ---- label ----
                cv2.putText(frame, f"{label}: {gesture}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If a hand is temporarily not tracked, reset its stroke state to avoid jumps.
        for side in ("Left", "Right"):
            if side not in seen_labels:
                prev_tip[side] = None
                prev_eraser_tip[side] = None
                prev_draw_pose[side] = False
                point_streak[side] = 0
                draw_hit_count[side] = 0
                draw_miss_count[side] = 0
                clear_hold[side] = 0
                tip_position_buffer[side].clear()  # Clear smoothing buffer

        # ---- HUD overlay: draw controls legend once per frame ----
        legend_lines = [
            "INDEX UP    = draw (stays on screen)",
            "OPEN PALM   = erase selected area",
            "THUMBS_DOWN = hold to clear canvas",
        ]
        legend_y0 = H - 52
        for li, txt in enumerate(legend_lines):
            cv2.putText(frame, txt, (12, legend_y0 + li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        trail_frames.append(cv2.GaussianBlur(frame, (5, 5), 0))

        display = square_view(frame, WINDOW_SIZE)
        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
