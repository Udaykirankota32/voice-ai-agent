# ─────────────────────────────────────────
# MEMORY SYSTEM
# Two levels:
# 1. Session memory - current conversation
# 2. Patient memory - across sessions
# ─────────────────────────────────────────


class SessionMemory:
	def __init__(self):
		# List of messages in this session
		self.history = []
		# Current detected language
		self.language = "en"
		# What the user is trying to do
		self.current_intent = None

	def add_user_message(self, message: str):
		"""Add what the user said"""
		self.history.append({"role": "user", "content": message})

	def add_agent_message(self, message: str):
		"""Add what the agent responded"""
		self.history.append({"role": "assistant", "content": message})

	def get_history(self) -> list:
		"""Get full conversation history for LLM"""
		return self.history

	def set_language(self, language: str):
		"""Remember which language user is speaking"""
		self.language = language

	def clear(self):
		"""Clear session when call ends"""
		self.history = []
		self.current_intent = None


class PatientMemory:
	def __init__(self):
		# patient_id -> patient data
		self.patients = {}

	def get_or_create_patient(self, name: str) -> dict:
		"""
		Find existing patient or create new one.
		In real system this queries the database.
		"""
		# Use name as simple ID for now
		patient_id = name.lower().replace(" ", "_")

		if patient_id not in self.patients:
			# New patient
			self.patients[patient_id] = {
				"id": patient_id,
				"name": name,
				"language": "en",  # preferred language
				"appointments": [],  # past appointments
				"last_doctor": None,  # last doctor visited
			}
			print(f"[MEMORY] New patient created: {name}")
		else:
			print(f"[MEMORY] Returning patient found: {name}")

		return self.patients[patient_id]

	def update_patient(self, name: str, data: dict):
		"""Update patient information"""
		patient_id = name.lower().replace(" ", "_")
		if patient_id in self.patients:
			self.patients[patient_id].update(data)

	def add_appointment(self, name: str, appointment: dict):
		"""Add appointment to patient history"""
		patient_id = name.lower().replace(" ", "_")
		if patient_id in self.patients:
			self.patients[patient_id]["appointments"].append(appointment)
			print(f"[MEMORY] Appointment saved for {name}: {appointment}")

	def get_patient_context(self, name: str) -> str:
		"""
		Returns a summary of patient history
		to inject into the LLM prompt so it
		remembers past interactions
		"""
		patient_id = name.lower().replace(" ", "_")
		if patient_id not in self.patients:
			return ""

		patient = self.patients[patient_id]
		context = f"Returning patient: {patient['name']}. "

		if patient["last_doctor"]:
			context += f"Last visited: Dr. {patient['last_doctor']}. "

		if patient["appointments"]:
			last = patient["appointments"][-1]
			context += f"Last appointment: {last.get('date', 'unknown')}. "

		return context


# ── GLOBAL INSTANCES ──
# One session memory per active call
# One patient memory shared across calls
session_memory = SessionMemory()
patient_memory = PatientMemory()


if __name__ == "__main__":
	print("Testing memory system...")

	# Test session memory
	session = SessionMemory()
	session.add_user_message("Book appointment with Dr. Sharma")
	session.add_agent_message("Sure, what date works for you?")
	session.add_user_message("Tomorrow at 10am")

	print("\nSession history:")
	for msg in session.get_history():
		print(f"  {msg['role']}: {msg['content']}")

	# Test patient memory
	patients = PatientMemory()
	patients.get_or_create_patient("Ravi Kumar")
	patients.add_appointment(
		"Ravi Kumar",
		{
			"doctor": "Dr. Sharma",
			"date": "2024-01-15",
			"time": "10:00 AM",
		},
	)

	# Simulate returning patient
	patients.get_or_create_patient("Ravi Kumar")
	context = patients.get_patient_context("Ravi Kumar")
	print(f"\nPatient context for LLM: {context}")

	print("\nMemory system working!")
