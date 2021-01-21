# Audio-Recognition
Recognizing more than 30 different audios.

Dataset Information:
	• Train and Test sets are generated from the dataset.
        • Train Set: Duration of sound varies from the categories to categories and also audio samples per category are maximum up to 250. Around ~10000 samples are taken to include all categories
        • Out of all around 30% data have received verification online whereas rest of the data unverified but having enough accuracy score of up to 70% which allows to use in model creation.
        • The size of test set is 1/5th of the training set data.

Solution Approach:
        • audio waveform are utiiized to create spectral features and artificial neural network approach is utilized to generate classification model.
        • Spectrogram of audio waveform will be utilized in the second portion to generate classification based on convolutional neural network to utilize inverse and rhythm features.
        • Python package to create to run this model across multiple platforms to classify audio
        without required to be trained across multiple platform.

Dependencies:
        • Please see the text file for the list of dependencies.

Notebook:
        • Run NN_audio_clasfc_sagar_soni.py in the choice of editor or from the CMD directly.

Librosa Install Requirements:
        • install_requires=[
                                'audioread >= 2.0.0',
                                'numpy >= 1.15.0',
                                'scipy >= 1.0.0',
                                'scikit-learn >= 0.14.0, != 0.19.0',
                                'joblib >= 0.14',
                                'decorator >= 3.0.0',
                                'resampy >= 0.2.2',
                                'numba >= 0.43.0',
                                'soundfile >= 0.9.0',
                                'pooch >= 1.0'
                        ]
        • Also required Tensorflow and Scikit-Learn on the system to run this py file.

