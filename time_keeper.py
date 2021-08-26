# Gelin Eguinosa Rosique

import time


class TimeKeeper:
    """
    Class to keep track of the time the programs expends processing the
    information. It doesn't count the time the program expends waiting for input
    from the user.
    """

    def __init__(self):
        """
        Create the basic variables to keep track of the running time of the
        program.
        """
        # Initialize variables
        self.start_time = time.time()
        self.runtime = 0
        # Set the state of the recording
        self.current_state = 'running'

    def pause(self):
        """
        Stop recording the time, until the user commands it to start recording
        again.
        """
        # Check if the TimeKeeper is currently running, otherwise we don't need
        # to anything.
        if self.current_state == 'running':
            # Update the total runtime of the program.
            self.runtime += time.time() - self.start_time
            # Update the state of the recording
            self.current_state = 'pause'

    def restart(self):
        """
        Resets the start time of the TimeKeeper, either if it is currently on
        pause or running. As a result, it continues recording if it was on
        pause, or resets the time if it was currently running.
        """
        # Reset the value of the start time
        self.start_time = time.time()
        # Update the state of the recording
        self.current_state = 'running'

    def run_time(self):
        """
        Transforms the elapsed time from the start of the program to a new format
        in hours, minutes and seconds.
        :return: A string containing the elapsed time in <hours:minutes:seconds>
        """        
        # Update the runtime value only is the TimeKeeper was running.
        if self.current_state == 'running':
            self.runtime += time.time() - self.start_time
            # Reset the start time, to avoid adding the same segment again
            self.start_time = time.time()

        # Calculating the hours, minutes, seconds and milliseconds
        hours = int(self.runtime / 3600)
        minutes = int((self.runtime - hours * 3600) / 60)
        seconds = int(self.runtime - hours * 3600 - minutes * 60)
        milliseconds = int((self.runtime - int(self.runtime)) * 1000)

        return f'{hours} h : {minutes} min : {seconds} sec : {milliseconds} mill'
