*Real-time Audio Recorder with Notifications*

-------------------------------------------------------------------------------

[Current Features]

-[Real-time Audio Recording]: The script captures audio from the default playback device hands free and saves it as '.wav' files in the specified output folder. The audio is recorded in a loop, overwriting the current file with each iteration.

-[Unique Naming Convention]: The audio files are named based on the year, month, date, and time, followed by the .wav format.

-[Status Notifications]: The program sends the following toast notifications to keep the user informed about the recording/app process:

	-"Recording in progress" when recording starts.
	-"Recording loop count" Loop 1,2 and so on.
	-"Recording process paused" when the user pauses recording.
	-"Recording process resumed" when the user resumes recording.
	-"Recording has ended" when the user stops the recording.
	-"App closed" Code exists successfully with exit code 0.
	-"Reset" when the user pauses the recording and hits reset, it should break the loop and start recording from loop count 1.
    	-"m clips from loop n being analyzed" where 'm' is the number of clips set by the user and 'n' is the loop number. (Displays files in the folder)
   	-"Analysis done" after each analysis phase is completed.

-[Looped Recording]: This application features looped recording, where it repeatedly records and overwrites the same file, making it useful for scenarios where you need the most recent audio data continuously.

-[Audio Analysis]: After each loop of recordings, these clips are sent for analysis. During this analysis, a message is displayed on the GUI and a toast notification is shown. Once the analysis is done, another message and notification are displayed.

-[Timer and Date/Time Display]: The application displays the elapsed time and current date/time while it is running.

-[Code Comments]: The code is extensively documented with comments, explaining its functionality and structure for better understanding and potential modifications.

[Modifications]

-The "Stop" button has been renamed to "Exit", without changing its functionality.
-The GUI status and toast notifications have been updated to reflect all stages of the recording and analysis process.
-A new function `analyze_clips` has been added to simulate the analysis of recorded audio clips. This function should be replaced with the actual analysis code.

