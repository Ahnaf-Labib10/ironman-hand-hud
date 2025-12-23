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

    # slightly strict avg_tip_dist so open palms don't look like fists
    if curled_fingers >= 3 and avg_tip_dist < 1.5 and thumb_close:
        return "FIST"

    if all([thumb_extended, index, middle, ring, pinky]):
        return "OPEN"

    if index and not any([middle, ring, pinky, thumb_extended]):
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
    """Target-style corner brackets."""
    # Top-left
    cv2.line(img, (x1, y1), (x1 + L, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + L), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - L, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + L), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + L, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - L), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - L, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - L), color, thickness)


def draw_waveform(img, origin, t, color):
    """Small animated waveform (purely visual flair)."""
    ox, oy = origin
    w = 110
    amp = 10
    pts = []
    for i in range(w):
        y = oy + int(math.sin((i * 0.14) + t * 3.2) * amp * 0.6 + math.sin((i * 0.05) + t * 2.0) * amp * 0.4)
        pts.append((ox + i, y))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2)


def draw_mini_bar_graph(img, origin, states, color):
    """Tiny bar graph for finger states (index/middle/ring/pinky)."""
    ox, oy = origin
    bar_w, gap = 6, 4
    max_h = 28
    for i, on in enumerate(states):
        h = max_h if on else 10
        x = ox + i * (bar_w + gap)
        cv2.rectangle(img, (x, oy - h), (x + bar_w, oy), color, -1)
        cv2.rectangle(img, (x, oy - h), (x + bar_w, oy), (255, 255, 255), 1)


def gesture_mode(gesture):
    """Turns gesture into a fun 'HUD mode' label."""
    return {
        "OPEN": "SCAN MODE",
        "FIST": "COMBAT MODE",
        "PEACE": "ANALYZE MODE",
        "POINT": "TARGET MODE",
        "THUMBS_UP": "CONFIRM",
        "THUMBS_DOWN": "DENY",
        "PINCH": "PINCH CTRL"
    }.get(gesture, "IDLE")


def mode_color(gesture):
    """Pick a theme color based on gesture/mode."""
    if gesture == "FIST":
        return (0, 0, 255)        # red
    if gesture == "OPEN":
        return (0, 255, 0)        # green
    if gesture == "PEACE":
        return (255, 255, 0)      # cyan-ish/yellow (BGR)
    if gesture in ("THUMBS_UP", "THUMBS_DOWN"):
        return (255, 200, 50)     # gold
    return (0, 255, 255)          # aqua


# ----------------------------
# Particles (simple physics)
# ----------------------------
def spawn_particles(particles, center, n=18):
    """Create a burst of particles around the hand."""
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
    """Move particles, fade them, draw them."""
    alive = []
    for p in particles:
        x, y, vx, vy, life, size = p
        x += vx
        y += vy
        vy += 0.06  # tiny gravity for a nicer “spark fall”
        life -= 1

        if life > 0:
            # brightness based on remaining life
            # (OpenCV doesn't have alpha here, so we fake fade by size)
            cv2.circle(frame, (int(x), int(y)), max(1, int(size * (life / 35))), (255, 255, 255), -1)
            alive.append([x, y, vx, vy, life, size])
    particles[:] = alive


# ----------------------------
# Main HUD v3
# ----------------------------
def draw_hud_v3(frame, center, bbox, hand_length, gesture, finger_states, t):
    """
    One function that draws the whole “Iron Man” look.
    We draw on an overlay and blend it for that transparent HUD feel.
    """
    x, y = center
    x1, y1, x2, y2 = bbox

    accent = mode_color(gesture)
    mode = gesture_mode(gesture)

    hud = frame.copy()

    # Brackets around hand
    draw_corner_brackets(hud, x1, y1, x2, y2, accent, thickness=2, L=20)

    # Scan line (moves up/down)
    scan = int((math.sin(t * 2.8) * 0.5 + 0.5) * (y2 - y1))
    scan_y = y1 + scan
    cv2.line(hud, (x1, scan_y), (x2, scan_y), (255, 255, 255), 1)

    # Dial with rotating tick marks
    radius = int(58 + hand_length * 15)
    rot = (t * 140) % 360

    for a in range(0, 360, 18):
        aa = math.radians(a + rot)
        r1 = radius + 5
        r2 = radius + (18 if a % 36 == 0 else 11)
        p1 = (x + int(math.cos(aa) * r1), y + int(math.sin(aa) * r1))
        p2 = (x + int(math.cos(aa) * r2), y + int(math.sin(aa) * r2))
        cv2.line(hud, p1, p2, accent, 2)

    # Rotating arc
    cv2.ellipse(hud, (x, y), (radius, radius), int(rot), 0, 255, accent, 2)
    cv2.circle(hud, (x, y), radius - 16, (0, 255, 255), 1)

    # Hand length progress
    prog = int(min(hand_length * 95, 300))
    cv2.ellipse(hud, (x, y), (radius - 22, radius - 22), 0, 0, prog, (0, 255, 0), 3)

    # Crosshair
    cv2.line(hud, (x - 16, y), (x + 16, y), (0, 255, 255), 1)
    cv2.line(hud, (x, y - 16), (x, y + 16), (0, 255, 255), 1)

    # Floating panel
    panel_w, panel_h = 220, 105
    px, py = x2 + 14, y1
    if px + panel_w > frame.shape[1]:
        px = x1 - panel_w - 14

    cv2.rectangle(hud, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.rectangle(hud, (px, py), (px + panel_w, py + panel_h), accent, 2)

    # Tech text
    cv2.putText(hud, f"{mode}", (px + 10, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(hud, f"LEN: {hand_length:.2f}", (px + 10, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    s1 = 100 + int(60 * math.sin(t * 1.6))
    s2 = 120 + int(40 * math.cos(t * 1.1))
    cv2.putText(hud, f"S1:{s1:03d}  S2:{s2:03d}", (px + 10, py + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Mini bar graph + waveform
    draw_mini_bar_graph(hud, (px + 12, py + panel_h + 35), finger_states, accent)
    draw_waveform(hud, (x - 55, y2 + 34), t, (0, 255, 255))

    # Blend overlay (transparent HUD feel)
    cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)


# ----------------------------
# Trail effect setup
# ----------------------------
# We keep some old frames, then blend them faintly on top.
trail_frames = deque(maxlen=6)

# Particle storage
particles = []

# Detect fist transitions (for particle bursts)
prev_gestures = {"Left": "UNKNOWN", "Right": "UNKNOWN"}

# Short history per hand to smooth noisy single-frame mis-detections
gesture_buffers = {
    "Left": deque(maxlen=5),
    "Right": deque(maxlen=5)
}

cap = cv2.VideoCapture(0)
start_time = time.time()

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        t = time.time() - start_time

        # ---- TRAIL: blend older frames very lightly ----
        # We add old frames from the deque onto the current frame
        # so motion looks “afterimaged” like a movie HUD.
        if trail_frames:
            trail = frame.copy()
            # Blend from oldest to newest; newest gets slightly stronger
            for k, old in enumerate(trail_frames):
                a = 0.06 + 0.02 * k
                cv2.addWeighted(old, a, trail, 1 - a, 0, trail)
            frame = trail

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ---- Update/draw particles (they sit on top of everything) ----
        update_and_draw_particles(frame, particles)

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label  # "Left" or "Right"

                points = np.array([[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark], dtype=np.float32)

                # Raw gesture for this frame
                raw_gesture = detect_gesture(points)
                hand_length = compute_hand_length(points)

                # --- Temporal smoothing of gesture ---
                buf = gesture_buffers[label]
                buf.append(raw_gesture)
                if buf:
                    # Most common gesture in the short history
                    smoothed = max(set(buf), key=buf.count)
                else:
                    smoothed = raw_gesture

                gesture = smoothed

                # Finger states for mini-graph (index/middle/ring/pinky)
                idx_state = finger_extended(points[8], points[6])
                mid_state = finger_extended(points[12], points[10])
                ring_state = finger_extended(points[16], points[14])
                pinky_state = finger_extended(points[20], points[18])
                finger_states = [idx_state, mid_state, ring_state, pinky_state]

                # Wrist center
                wx, wy = int(points[0][0]), int(points[0][1])

                # Hand bounding box (for brackets)
                x1 = int(np.min(points[:, 0]) - 10)
                y1 = int(np.min(points[:, 1]) - 10)
                x2 = int(np.max(points[:, 0]) + 10)
                y2 = int(np.max(points[:, 1]) + 10)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                # Skeleton (optional, but helps while testing)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ---- Particle burst when you *enter* FIST (based on smoothed gesture) ----
                if prev_gestures.get(label, "UNKNOWN") != "FIST" and gesture == "FIST":
                    spawn_particles(particles, (wx, wy), n=22)

                prev_gestures[label] = gesture

                # Draw full HUD
                draw_hud_v3(frame, (wx, wy), (x1, y1, x2, y2), hand_length, gesture, finger_states, t)

                # Small label
                cv2.putText(frame, f"{label}: {gesture}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Store this frame for trail (store a slightly blurred copy for nicer trails)
        trail_frames.append(cv2.GaussianBlur(frame, (5, 5), 0))

        cv2.imshow("Iron Man Hand HUD v3", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
