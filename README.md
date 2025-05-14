🎓 Classroom Behavior Recognition using CNN & Streamlit

An automated, intelligent system for real-time recognition of student behavior in classrooms using Convolutional Neural Networks (CNNs) and deployed through a Streamlit web app. The system allows users (teachers, administrators) to upload images  from the classroom and instantly classifies student behavior (e.g., attentive, distracted, sleeping), enabling timely interventions to improve engagement and discipline.

🚀 Features

✅ Image-based student behavior classification
🧠 CNN-based deep learning model with high accuracy
🗃️ Image preprocessing using Keras’ ImageDataGenerator
📊 Performance visualization through confusion matrix and training plots
💾 Model saving in HDF5 format for future inference
📈 Classification metrics (accuracy, precision, recall, F1-score)
⚙️ Real-time prediction via Streamlit interface

🛠️ Tech Stack

Python 3.x
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Streamlit
🗂️ How to Use

1. Setup Environment
Install required packages:

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit
2. Prepare Dataset
Place your dataset zip file (e.g., classroom_behavior.zip) in the working directory.
Extract it and ensure folders are organized by class names:
attentive/
distracted/
sleeping/
etc.
3. Train the Model
Run your training script to:

Load and preprocess the data
Build & train the CNN model
Save the trained model as behavior_model.h5
4. Run Streamlit App
streamlit run app.py
5. Upload and Predict
Upload a classroom image or video frame
Get real-time prediction of student behavior along with the confidence level
👥 Team Members

[Daksh Kumar]
[Yash Sharma]
[Yug Bneniwal]
🎓 Guided By

[Dr. Roshi Saxena]
Xebia
🤝 Acknowledgements

Kaggle – Student behavior datasets and community resources
TensorFlow – Deep Learning Framework
Streamlit – App interface for real-time inference
