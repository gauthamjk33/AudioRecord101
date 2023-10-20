import os
import time
import threading
from datetime import datetime
from plyer import notification
import soundcard as sc
import soundfile as sf
import PySimpleGUI as sg

# Function to delete old audio files in the specified output folder
def delete_old_audio_files(output_folder):
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Deletes the file
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

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
        print("An error occurred:", str(e))

    # Wait here if pause event is set
    while pause_event.is_set():
        time.sleep(1)

# State changes to analysing the recorded audio clips
def analyze_clips(loop_number, num_clips):
    # Add the analysis code here (model)
    print(f"Analyzing {num_clips} clips from loop {loop_number}")
    time.sleep(15)  # Simulate time taken for analysis
    print("Analysis done")

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
            title="Audio Recording Status",
            message=f"Loop number {loop_number} completed",
            app_name="MyAudioApp"
        )

        window['status'].update(f'Recordings from loop {loop_number} analyzing', text_color='yellow')
        notification.notify(
            title="Audio Recording Status",
            message=f"{num_clips} clips from loop {loop_number} being analyzed",
            app_name="MyAudioApp"
        )

        analyze_clips(loop_number, num_clips)

        window['status'].update('Analysis done', text_color='green')
        notification.notify(
            title="Audio Recording Status",
            message="Analysis done",
            app_name="MyAudioApp"
        )

        loop_number += 1

output_folder = r'D:/Realtimerecording'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

samplerate = 48000
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

layout = [
    [sg.Text("Recording Status", text_color='green', font=('Helvetica', 18), key='status')],
    [sg.Text('', size=(40, 2), key='timer', justification='right')],
    [sg.Button("Start", size=(7, 1)), sg.Button("Pause", size=(7, 1)), sg.Button("Resume", size=(7, 1)),
     sg.Button("Exit", size=(7, 1)), sg.Button("Reset", size=(7, 1)), sg.Button("Edit", size=(7, 1))],
     [sg.Text(f'Clips set to {num_clips} clips', key='clip_count')]
]

window = sg.Window("Fake Voice Alert", layout, finalize=True, return_keyboard_events=True,
                   location=(150, 150), size=(500, 250))
start_time = datetime.now()

recording_thread_started = False
pause_event = threading.Event()

while True:
    event, values = window.read(timeout=1000)

    if event == sg.WIN_CLOSED or event == 'Exit':
        window['status'].update('Recording has ended', text_color='red')
        time.sleep(5)
        window['status'].update('Analysing the recorded clips', text_color='red')

        # Show a toast notification when application is closed
        notification.notify(
            title="Application Status",
            message="Application closed",
            app_name="MyAudioApp"
        )
        break

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
            title="Audio Recording Status",
            message="Recording process paused",
            app_name="MyAudioApp"
        )

        pause_event.set()

    if event == 'Resume':
        window['status'].update('Recording in progress', text_color='green')

        # Show a toast notification when recording is resumed
        notification.notify(
            title="Audio Recording Status",
            message="Recording process resumed",
            app_name="MyAudioApp"
        )

        pause_event.clear()

    if event == 'Reset':
        window['status'].update('Resetting...', text_color='red')
        notification.notify(
            title="Audio Recording Status",
            message="Application reset successful",
            app_name="MyAudioApp"
        )
        recording_thread_started = False
        pause_event.set()
        delete_old_audio_files(output_folder)
        window['status'].update('Ready', text_color='green')
        pause_event.clear()

    #Edit the number of clips
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
                        title="Audio Recording Status",
                        message="Changes saved",
                        app_name="MyAudioApp"
                    )
                except ValueError:
                    sg.popup('Please enter an integer value')
                break
        window_edit.close()

    # Call update_timer and update the timer field
    window['timer'].update(update_timer(start_time))

window.close()
