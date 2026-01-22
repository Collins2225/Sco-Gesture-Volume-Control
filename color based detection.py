import cv2
import numpy as np


# -------------------- Constants (remove repeated 255) --------------------
MAX = 255
H_MAX = 180

WHITE = (MAX, MAX, MAX)
BLACK = (0, 0, 0)

BGR_RED = (0, 0, MAX)
BGR_GREEN = (0, MAX, 0)
BGR_BLUE = (MAX, 0, 0)
BGR_YELLOW = (0, MAX, MAX)
BGR_GRAY = (128, 128, 128)


def hsv_bounds(low, high):
    """Helper: convert tuples to HSV numpy arrays."""
    return np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)


class ColorObjectDetector:
    def __init__(self):
        # Each color can have 1 or more HSV ranges (red needs 2 because hue wraps)
        self.color_specs = {
            "Red": {
                "ranges": [
                    hsv_bounds((0, 120, 70), (10, MAX, MAX)),
                    hsv_bounds((170, 120, 70), (H_MAX, MAX, MAX)),
                ],
                "bgr": BGR_RED,
            },
            "Green": {
                "ranges": [hsv_bounds((40, 50, 50), (80, MAX, MAX))],
                "bgr": BGR_GREEN,
            },
            "Blue": {
                "ranges": [hsv_bounds((100, 100, 70), (130, MAX, MAX))],
                "bgr": BGR_BLUE,
            },
            "Yellow": {
                "ranges": [hsv_bounds((20, 100, 100), (30, MAX, MAX))],
                "bgr": BGR_YELLOW,
            },
            "White": {
                "ranges": [hsv_bounds((0, 0, 200), (H_MAX, 30, MAX))],
                "bgr": WHITE,
            },
            "Black": {
                "ranges": [hsv_bounds((0, 0, 0), (H_MAX, MAX, 50))],
                "bgr": BLACK,
            },
            "Gray": {
                "ranges": [hsv_bounds((0, 0, 50), (H_MAX, 50, 200))],
                "bgr": BGR_GRAY,
            },
        }

        self.min_area = 500
        self.detected_objects = []

        # Precompute kernel once (faster + cleaner)
        self.kernel = np.ones((5, 5), np.uint8)

        # Optional: if True, keep only the largest contour per color
        self.largest_only = False

    def preprocess_frame(self, frame):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def create_mask(self, hsv_frame, ranges):
        """Create one final mask from 1..N HSV ranges."""
        mask_total = None
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        # Clean mask: OPEN removes small noise, CLOSE fills holes
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask_total

    @staticmethod
    def find_contours(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def get_bounding_box(contour):
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        return x, y, w, h, cx, cy

    def detect_objects(self, frame):
        self.detected_objects = []
        hsv = self.preprocess_frame(frame)

        for color_name, spec in self.color_specs.items():
            mask = self.create_mask(hsv, spec["ranges"])
            contours = self.find_contours(mask)

            if not contours:
                continue

            # If robotics-style: keep only the biggest contour
            if self.largest_only:
                contours = [max(contours, key=cv2.contourArea)]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= self.min_area:
                    continue

                x, y, w, h, cx, cy = self.get_bounding_box(contour)
                self.detected_objects.append({
                    "color": color_name,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": area,
                    "contour": contour,
                    "draw_color": spec["bgr"],
                })

        return self.detected_objects

    def draw_detections(self, frame):
        out = frame.copy()

        for obj in self.detected_objects:
            x, y, w, h = obj["bbox"]
            cx, cy = obj["center"]
            color = obj["draw_color"]

            cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
            cv2.circle(out, (cx, cy), 5, color, -1)

            label = f"{obj['color']} ({int(obj['area'])})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y - 30), (x + tw + 10, y), color, -1)
            cv2.putText(out, label, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            # Crosshair
            cv2.line(out, (cx - 15, cy), (cx + 15, cy), color, 2)
            cv2.line(out, (cx, cy - 15), (cx, cy + 15), color, 2)

        return out

    def get_robotics_commands(self):
        commands = []
        for obj in self.detected_objects:
            cx, cy = obj["center"]
            commands.append({
                "action": "PICKUP",
                "target_color": obj["color"],
                "position_x": cx,
                "position_y": cy,
                "confidence": "HIGH" if obj["area"] > 2000 else "MEDIUM",
            })
        return commands


class TrafficLightDetector:
    def __init__(self):
        self.detector = ColorObjectDetector()
        self.light_state = "UNKNOWN"

        self.state_colors = {
            "STOP": BGR_RED,
            "CAUTION": BGR_YELLOW,
            "GO": BGR_GREEN,
            "NO LIGHT DETECTED": BGR_GRAY,
        }

    def detect_traffic_light(self, frame):
        objects = self.detector.detect_objects(frame)
        colors = {obj["color"] for obj in objects}

        if "Red" in colors:
            self.light_state = "STOP"
        elif "Yellow" in colors:
            self.light_state = "CAUTION"
        elif "Green" in colors:
            self.light_state = "GO"
        else:
            self.light_state = "NO LIGHT DETECTED"

        return self.light_state, objects


def main():
    detector = ColorObjectDetector()
    traffic_detector = TrafficLightDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode = "object"
    show_info = False
    show_commands = False

    print("\n" + "=" * 60)
    print("COLOR-BASED OBJECT DETECTION SYSTEM")
    print("=" * 60)
    print("Keys: o=Object  t=Traffic  i=Info  r=Robot  l=LargestOnly  q=Quit")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        frame = cv2.flip(frame, 1)

        if mode == "object":
            objects = detector.detect_objects(frame)
            display = detector.draw_detections(frame)

            cv2.putText(display, f"MODE: Object | Objects: {len(objects)} | LargestOnly: {detector.largest_only}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLACK, 2)

            if show_info and objects:
                y = 60
                for obj in objects:
                    txt = f"{obj['color']} center={obj['center']} area={int(obj['area'])}"
                    cv2.putText(display, txt, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
                    y += 22

            if show_commands and objects:
                cmds = detector.get_robotics_commands()
                y = 60
                cv2.putText(display, "ROBOT COMMANDS:", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR_BLUE, 2)
                y += 28
                for c in cmds:
                    txt = f"PICKUP {c['target_color']} at ({c['position_x']}, {c['position_y']}) [{c['confidence']}]"
                    cv2.putText(display, txt, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
                    y += 22

        else:
            light_state, _ = traffic_detector.detect_traffic_light(frame)
            display = traffic_detector.detector.draw_detections(frame)

            color = traffic_detector.state_colors.get(light_state, WHITE)
            cv2.rectangle(display, (10, 60), (300, 120), color, -1)
            cv2.putText(display, light_state, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3)

            cv2.putText(display, "MODE: Traffic Light",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)

        cv2.putText(display, "Keys: o Object | t Traffic | i Info | r Robot | l LargestOnly | q Quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        cv2.imshow("Color Object Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            mode = "object"
            show_info = False
            show_commands = False
            print("[Mode] Object")
        elif key == ord("t"):
            mode = "traffic"
            show_info = False
            show_commands = False
            print("[Mode] Traffic")
        elif key == ord("i"):
            show_info = not show_info
            print(f"[Info] {'ON' if show_info else 'OFF'}")
        elif key == ord("r"):
            show_commands = not show_commands
            print(f"[Robot] {'ON' if show_commands else 'OFF'}")
        elif key == ord("l"):
            detector.largest_only = not detector.largest_only
            traffic_detector.detector.largest_only = detector.largest_only
            print(f"[LargestOnly] {'ON' if detector.largest_only else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
