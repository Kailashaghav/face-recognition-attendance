Face Recognition Attendance System

🎯 A real-time AI-powered attendance system using Computer Vision & Machine Learning

📌 Overview

The Face Recognition Attendance System automates attendance tracking by detecting and recognizing faces through a webcam. It uses a K-Nearest Neighbors (KNN) model for classification and provides a live dashboard for monitoring attendance records.


✨ Key Highlights
	•	🎥 Real-time face detection & recognition
	•	🧠 Machine Learning (KNN Classifier)
	•	📊 Live dashboard using Streamlit
	•	📝 Automatic attendance with timestamp
	•	🚫 Prevents duplicate entries
	•	🔊 Voice feedback system
	•	⚡ Fast & lightweight implementation


🛠️ Tech Stack

Technology	Usage
Python	Core programming
OpenCV	Face detection
Scikit-learn	KNN model
NumPy	Data processing
Pandas	Data handling
Streamlit	Dashboard UI


📂 Project Structure

FACE RECOGNITION/
│── data/
│   ├── faces_data.pkl
│   ├── names.pkl
│   ├── haarcascade_frontalface_default.xml
│
│── Attendance/
│   ├── Attendance_dd-mm-yyyy.csv
│
│── test.py        # Face recognition system
│── app.py         # Streamlit dashboard
│── README.md


⚙️ Installation & Setup

🔹 Clone Repository

git clone https://github.com/Kailashaghav/face-recognition-attendance.git
cd face-recognition-attendance

🔹 Install Dependencies

pip install opencv-python scikit-learn numpy pandas streamlit pyttsx3



▶️ Run the Project

📸 Start Face Recognition

python3 test.py

	•	Press O → Mark Attendance
	•	Press Q → Quit


📊 Launch Dashboard

streamlit run app.py


📊 Output
	•	Attendance saved in:

Attendance/Attendance_dd-mm-yyyy.csv

	•	Dashboard displays:
	•	✅ Live attendance records
	•	✅ Latest entry highlight


🧠 How It Works
	1.	Capture face using webcam
	2.	Detect face using Haar Cascade
	3.	Convert image into feature vector
	4.	Predict identity using KNN
	5.	Store name + timestamp in CSV
	6.	Display data in Streamlit dashboard


📸 Demo Preview

🔥 Add screenshots or demo video here for best impact

🚀 Future Enhancements
	•	🔄 Fully automatic attendance (no key press)
	•	📱 Mobile/web camera integration
	•	📊 Advanced analytics & graphs
	•	☁️ Cloud database (Firebase/MongoDB)
	•	🔐 Face recognition accuracy improvement


💼 Why This Project Matters

This project demonstrates:
	•	Real-world application of Computer Vision
	•	Hands-on implementation of Machine Learning
	•	Integration of Backend + Frontend (Streamlit)
	•	Practical automation solution for attendance systems


👨‍💻 Author

Kailash Aghav

⭐ Show Your Support

If you like this project, please ⭐ star the repo and share it!
