from services.config.workout_config import PROMPT


class LLMCoach:
    def __init__(self, groq_client=None):
        self.client = groq_client
        self.history = []
        self.system_prompt = PROMPT

    def _local_feedback(self, event, issue):
        if event == "workout_started":
            return "Start strong. Keep your core tight and move with control."

        if event == "workout_completed":
            return "Workout complete. Great work. Recover, hydrate, and come back stronger."

        if event == "set_completed":
            return "Good set. Reset your posture and prepare for the next one."

        if event == "no_pose_detected":
            return "Step back into frame and face the camera clearly."

        if issue:
            return issue

        return "Good form. Keep breathing and stay controlled."

    def give_feedback(self, event, issue):
        prompt = f"Event: {event}"

        if issue:
            prompt += f" Form Issue: {issue}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-10:],
            {"role": "user", "content": prompt}
        ]

        if self.client is None:
            text = self._local_feedback(event, issue)
            self.history.append({"role": "assistant", "content": text})
            return text

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4,
            )

            text = response.choices[0].message.content.strip()
        except Exception:
            text = self._local_feedback(event, issue)

        self.history.append({"role": "assistant", "content": text})

        return text
    