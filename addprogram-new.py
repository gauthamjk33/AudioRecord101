import os
import sys
import time
import threading
from datetime import datetime
from plyer import notification
import soundcard as sc
import soundfile as sf
import PySimpleGUI as sg
import PyInstaller

# IMPORTS FOR MODEL PREDICTION
import librosa
import tensorflow as tf
import numpy as np


# Function to delete old audio files in the specified output folder
def delete_old_audio_files(output_folder):
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Deletes the file
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            print("Delete Old Audio files")


# Function to record audio from the default playback device and save it as .wav files in the output folder
def record_audio(output_folder, samplerate, record_sec, clip_number, pause_event):
    try:
        default_playback_device_id = str(sc.default_speaker().name)

        # Generate a unique identifier based on the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')

        # Create a filename with the unique identifier and clip number
        output_file_name = os.path.join(output_folder, f'clip_{clip_number}_{timestamp}.wav')

        with sc.get_microphone(id=default_playback_device_id, include_loopback=True).recorder(
                samplerate=samplerate) as mic:
            data = mic.record(numframes=samplerate * record_sec)
            data = data[:, 0]

            sf.write(file=output_file_name, data=data, samplerate=samplerate)

            print("Audio clip saved successfully to:", output_file_name)

    except Exception as e:
        print(e)
        print("An error occurred:", str(e))
        print("Record Files")

    # Wait here if pause event is set
    while pause_event.is_set():
        time.sleep(1)


# FUNCTIONS FOR PREPROCESSING AND PREDICTION

# Preprocessing Audio Files
def preprocess_data(file_path, max_time_steps=109, sample_rate=22050, duration=3, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram


# Make Predictions
def makePrediction(audio_file_path, model_file, max_time_steps):
    mel_spectrogram = preprocess_data(audio_file_path, max_time_steps=max_time_steps)

    #reshaping the spectrogram
    input_data = np.expand_dims(mel_spectrogram, axis=0)

    result = ''
    threshold = 0.377089

    # Predict using the loaded classifier
    prediction = model_file.predict(input_data)

    # Convert the prediction to a human-readable label
    if (prediction[0][1] > threshold):
        result = "BONAFIDE"
    else:
        result = "SPOOF"

    return result


# State changes to analysing the recorded audio clips

# Updated Function with model loading and prediction

def analyze_clips(loop_number, num_clips, directory):
    # Load the pre-trained model
    model = tf.keras.models.load_model('basic_resnet.h5')
    print(f"Analyzing {num_clips} clips from loop {loop_number}")
    audio_samples = []
    bonafide_count = 0
    spoof_count = 0

    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = f"{directory}/{filename}"
            result = makePrediction(file_path, model, max_time_steps=109)
            if result == "BONAFIDE":
                bonafide_count += 1
            else:
                spoof_count += 1
            print(f"Audio file is {result}")
            audio_samples.append(filename)

    # Determine the majority result after processing all files
    if bonafide_count > spoof_count:
        majority_result = "BONAFIDE"
    elif spoof_count > bonafide_count:
        majority_result = 'SPOOFED'
    else:
        majority_result = "INCONCLUSIVE"

    # Display the result in a notification and in the log
    notification.notify(
        title="Audio Analysis Result: Majority Result",
        message=f"Majority of audio files in loop {loop_number} are {majority_result}",
        app_name="MyAudioApp1"
    )
    print(f"Majority of audio files in loop {loop_number} are {majority_result}")

    time.sleep(15)  # Simulate time taken for analysis
    print(f"Analysis done")
    return majority_result

# Work Left ----> Getting the Result as notification

def count_audio_files(output_folder):
    audio_files = [filename for filename in os.listdir(output_folder) if filename.endswith('.wav')]
    return len(audio_files)


def record_audio_batch(output_folder, samplerate, record_sec, num_clips, pause_event):
    loop_number = 1
    while True:
        delete_old_audio_files(output_folder)  # Delete old files before starting a new loop

        for clip_number in range(1, num_clips + 1):
            if pause_event.is_set():
                break
            record_audio(output_folder, samplerate, record_sec, clip_number, pause_event)
            time.sleep(1)  # Sleep for 1 second between recordings

        if pause_event.is_set():
            break

        notification.notify(
            title="Audio Recording Status: Recording Completed",
            message=f"Loop number {loop_number} completed",
            app_name="MyAudioApp1"
        )

        window['status'].update(f'Recordings from loop {loop_number} analyzing', text_color='yellow')
        notification.notify(
            title="Audio Recording Status: Analyzing",
            message=f"{num_clips} clips from loop {loop_number} being analyzed",
            app_name="MyAudioApp1"
        )

        majority_result= analyze_clips(loop_number, num_clips, output_folder)

        window['status'].update('Analysis done', text_color='green')
        notification.notify(
            title="Audio Recording Status: Analysis Complete",
            message="Analysis done",
            app_name="MyAudioApp1"

        )

        num_audio_files = count_audio_files(output_folder)

        # Display the result notification with the number of audio files
        notification.notify(
            title="Analysis Result: Complete Loop",
            message=f"Analysis for loop {loop_number} is complete. {num_audio_files} audio files recorded. Majority of audio files are {majority_result}",
            app_name="MyAudioApp1"
        )
        num_audio_files = count_audio_files(output_folder)
        print(
            f"Number of audio files in recorded in loop {loop_number}: {num_audio_files}. Majority of audio files are {majority_result}")

        loop_number += 1


output_folder = os.path.expanduser('~/MyAudioApp/Audio1')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

samplerate = 22050
record_sec = 5
num_clips = 5


# Define the update_timer function
def update_timer(start_time):
    current_time = datetime.now()
    elapsed_time = current_time - start_time
    elapsed_seconds = int(elapsed_time.total_seconds())
    formatted_time = f"Time Open: {elapsed_seconds} seconds"
    current_datetime = current_time.strftime("Date: %Y-%m-%d Time: %H:%M:%S")
    return f"{formatted_time}\n{current_datetime}"

# Login screen layout

window_title = "Fake Voice Alert"
rediminds_logo = "RM-White-Transparent-Logo.png"
copyright_text = "Copyright © 2023 RediMinds, Inc. All rights reserved."

login_layout = [
    [sg.Image(rediminds_logo, size=(400, 200))],
    [sg.Text('Username:'), sg.Input(size=(25, 1),key='-USERNAME-', enable_events=True)],
    [sg.Text('Password:'), sg.Input(size=(25, 1),key='-PASSWORD-', password_char='*', enable_events=True)],
    [sg.Button('Login')],
    [sg.Text('', key='-LOGIN_MESSAGE-', text_color='red')],
    [sg.Text(copyright_text, justification='center', text_color='white')],
]

# Create the login window
login_window = sg.Window(window_title, login_layout)
# Login credentials
username = 'Admin'
password = 'Admin@123'

while True:
    event_login, values_login = login_window.read()

    if event_login == sg.WINDOW_CLOSED:
        sys.exit(0)  # Break the loop first

    if event_login == 'Login':
        if values_login['-USERNAME-'] == username and values_login['-PASSWORD-'] == password:
            notification.notify(
                title="Login Status: Successful",
                message="Login Successful, Welcome Administrator",
                app_name="MyAudioApp1"
            )
            break  # Break the loop when login is successful
        else:
            login_window['-LOGIN_MESSAGE-'].update('Login Credentials Invalid')
            notification.notify(
                title="Login Status: Failed",
                message="Login attempt failed",
                app_name="MyAudioApp1"
            )

login_window.close()  # Close the window after breaking the loop

if event_login == sg.WINDOW_CLOSED:  # If the window was closed without logging in
    sys.exit(0)  # Exit the program


#Continued GUI
window_title = "Fake Voice Alert"
rediminds_logo = "RM-White-Transparent-Logo.png"
copyright_text = "Copyright © 2023 RediMinds, Inc. All rights reserved."
print_output = []
layout = [
    [sg.Image(rediminds_logo, size=(400, 200))],
    [sg.Text("Recording Status", text_color='green', font=('Helvetica', 18), key='status')],
    [sg.Text('', size=(40, 2), key='timer', justification='right')],
    [sg.Button("Start", size=(7, 1)), sg.Button("Pause", size=(7, 1)), sg.Button("Resume", size=(7, 1)),
     sg.Button("Exit", size=(7, 1)), sg.Button("Reset", size=(7, 1)), sg.Button("Edit", size=(7, 1))],
    [sg.Text(f'Clips set to {num_clips} clips', key='clip_count')],
    [sg.Text(copyright_text, justification='center', text_color='white')],
    [sg.Multiline(default_text='', size=(50, 15), key='-OUTPUT-', enable_events=True, reroute_cprint=True, autoscroll=True, disabled=True)],

]

window = sg.Window(window_title, layout, finalize=True, return_keyboard_events=True,
                   location=(150, 300), size=(600, 600))
start_time = datetime.now()

recording_thread_started = False
pause_event = threading.Event()
# Redirect print statements to the output element
def custom_print(*args, **kwargs):
    text = ' '.join(map(str, args))
    print_output.append(text)
    window['-OUTPUT-'].update('\n'.join(print_output))

print = custom_print

exit_confirmed = False
while True:
    event, values = window.read(timeout=1000)

    if event == sg.WINDOW_CLOSED and not exit_confirmed:
        if sg.popup_yes_no('Are you sure you want to exit?') == 'Yes':
            sys.exit(0)
        else:
            continue

    if event == sg.WIN_CLOSED or event == 'Exit':
        window['status'].update('Recording has ended', text_color='red')
        time.sleep(5)
        window['status'].update('Analysing the recorded clips', text_color='red')

        # Show a toast notification when application is closed
        notification.notify(
            title="Application Status: Close",
            message="Application closed",
            app_name="MyAudioApp1"
        )
        sys.exit(0)

    if event == 'Start' and not recording_thread_started:
        window['status'].update('Recording in progress', text_color='green')
        recording_thread_started = True
        recording_thread = threading.Thread(target=record_audio_batch,
                                            args=(output_folder, samplerate, record_sec, num_clips, pause_event))
        recording_thread.daemon = True
        recording_thread.start()

    if event == 'Pause':
        window['status'].update('Recording process paused', text_color='red')

        # Show a toast notification when recording is paused
        notification.notify(
            title="Audio Recording Status: Paused",
            message="Recording process paused",
            app_name="MyAudioApp1"
        )

        pause_event.set()

    if event == 'Resume':
        window['status'].update('Recording in progress', text_color='green')

        # Show a toast notification when recording is resumed
        notification.notify(
            title="Audio Recording Status: Resumed",
            message="Recording process resumed",
            app_name="MyAudioApp1"
        )

        pause_event.clear()

    if event == 'Reset':
        window['status'].update('Resetting...', text_color='red')
        notification.notify(
            title="Audio Recording Status: Reset",
            message="Application reset successful",
            app_name="MyAudioApp1"
        )
        recording_thread_started = False
        pause_event.set()
        delete_old_audio_files(output_folder)
        window['status'].update('Ready', text_color='green')
        pause_event.clear()

    # Edit the number of clips
    if event == 'Edit':
        layout_edit = [[sg.Text('Enter number of clips:'), sg.Input(key='-IN-', enable_events=True)],
                       [sg.Button('Save')]]
        window_edit = sg.Window('Edit Number of Clips', layout_edit)
        while True:
            event_edit, values_edit = window_edit.read()
            if event_edit == sg.WINDOW_CLOSED or event_edit == 'Save':
                try:
                    num_clips = int(values_edit['-IN-'])
                    window['clip_count'].update(f'Clips set to {num_clips} clips')
                    notification.notify(
                        title="Audio Recording Status: Edit Clips",
                        message="Changes saved",
                        app_name="MyAudioApp1"
                    )
                except ValueError:
                    sg.popup('Please enter an integer value')
                break
        window_edit.close()

    # Call update_timer and update the timer field
    window['timer'].update(update_timer(start_time))

window.close()
