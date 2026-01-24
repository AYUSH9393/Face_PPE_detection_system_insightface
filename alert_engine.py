import time
from datetime import datetime
from typing import Dict
import os

# Optional deps
try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    from twilio.rest import Client
except Exception:
    Client = None


class AlertEngine:
    """
    Central alert dispatcher
    - Reads config from DB
    - Applies cooldown
    - Triggers buzzer / WhatsApp
    """

    def __init__(self, db):
        self.db = db
        self.last_alert = {}  # key -> timestamp
        self._load_config()

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    def _load_config(self):
        cfg = self.db.system_config.find_one(
            {"config_type": "alerts"}
        ) or {}

        self.enabled = cfg.get("enabled", True)
        self.channels = cfg.get("channels", ["console"])
        self.cooldown_sec = cfg.get("cooldown_sec", 30)

        self.whatsapp_cfg = cfg.get("whatsapp", {})
        self.buzzer_cfg = cfg.get("buzzer", {})

    def reload(self):
        self._load_config()
        print("🔔 Alert config reloaded")

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def handle_violation(self, violation: Dict):
        """
        Called by PPE violation logger
        """
        if not self.enabled:
            return

        key = f"{violation.get('camera_id')}:{violation.get('person_id')}"

        now = time.time()
        last = self.last_alert.get(key, 0)

        if now - last < self.cooldown_sec:
            return  # cooldown active

        self.last_alert[key] = now

        message = self._build_message(violation)

        if "buzzer" in self.channels:
            self._trigger_buzzer()

        if "whatsapp" in self.channels:
            self._send_whatsapp(message)

        print("🚨 Alert triggered:", message)

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _build_message(self, v: Dict) -> str:
        return (
            f"⚠️ PPE VIOLATION\n"
            f"Camera: {v.get('camera_id')}\n"
            f"Person: {v.get('person_name', 'Unknown')}\n"
            f"Missing: {', '.join(v.get('missing_ppe', []))}\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

    # --------------------------------------------------
    # BUZZER
    # --------------------------------------------------
    def _trigger_buzzer(self):
        sound = self.buzzer_cfg.get("sound", "alert.wav")

        if playsound and os.path.exists(sound):
            try:
                playsound(sound, block=False)
            except Exception as e:
                print("🔔 Buzzer error:", e)
        else:
            print("🔔 BUZZER (fallback)")

    # --------------------------------------------------
    # WHATSAPP
    # --------------------------------------------------
    # --------------------------------------------------
    # WHATSAPP
    # --------------------------------------------------
    def _send_whatsapp(self, message: str):
        if not Client:
            print("📲 WhatsApp skipped (twilio not installed)")
            return

        sid = self.whatsapp_cfg.get("sid")
        token = self.whatsapp_cfg.get("token")
        from_no = self.whatsapp_cfg.get("from")
        to_no = self.whatsapp_cfg.get("to")

        if not all([sid, token, from_no, to_no]):
            print("📲 WhatsApp config incomplete")
            return

        def _send():
            try:
                client = Client(sid, token)
                client.messages.create(
                    body=message,
                    from_=f"whatsapp:{from_no}",
                    to=f"whatsapp:{to_no}"
                )
                print("📲 WhatsApp alert sent")
            except Exception as e:
                print("📲 WhatsApp error:", e)

        # Run in separate thread to avoid blocking main loop
        import threading
        t = threading.Thread(target=_send, daemon=True)
        t.start()

