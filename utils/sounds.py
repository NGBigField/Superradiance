import sys 
import winsound
from time import sleep

def ascend() -> None:
    for n in range(12):
        beep(note=n, duration=0.2)
        sleep(0.01)

def alarm() -> None:
    for _ in range(5):
        beep(duration=0.5)

def beep(
    duration:float=0.400, # sec 
    note:int=0,
    octave:int=5,
) -> None:  
    freq = note_freq(note=note, octave=octave)
    if sys.platform == 'win32':
        winsound.Beep(
            frequency=int(freq), 
            duration=int(duration*1000)  # In mili-sec
        )
    else:
        sys.stdout.write("\a")


NOTES_TABLE = [
#OCTAVE: 0	 1	     2	      3	      4	      5	       6	   7	    8         Note:
    [16.35,	32.70,	65.41 ,	130.81,	261.63,	523.25,	1046.50, 2093.00, 4186.01], # C	    
    [17.32,	34.65,	69.30 ,	138.59,	277.18,	554.37,	1108.73, 2217.46, 4434.92], # C#/Db	
    [18.35,	36.71,	73.42 ,	146.83,	293.66,	587.33,	1174.66, 2349.32, 4698.63], # D	    
    [19.45,	38.89,	77.78 ,	155.56,	311.13,	622.25,	1244.51, 2489.02, 4978.03], # D#/Eb	
    [20.60,	41.20,	82.41 ,	164.81,	329.63,	659.25,	1318.51, 2637.02, 5274.04], # E	    
    [21.83,	43.65,	87.31 ,	174.61,	349.23,	698.46,	1396.91, 2793.83, 5587.65], # F	    
    [23.12,	46.25,	92.50 ,	185.00,	369.99,	739.99,	1479.98, 2959.96, 5919.91], # F#/Gb	
    [24.50,	49.00,	98.00 ,	196.00,	392.00,	783.99,	1567.98, 3135.96, 6271.93], # G	    
    [25.96,	51.91,	103.83,	207.65,	415.30,	830.61,	1661.22, 3322.44, 6644.88], # G#/Ab	
    [27.50,	55.00,	110.00,	220.00,	440.00,	880.00,	1760.00, 3520.00, 7040.00], # A	    
    [29.14,	58.27,	116.54,	233.08,	466.16,	932.33,	1864.66, 3729.31, 7458.62], # A#/Bb	
    [30.87,	61.74,	123.47,	246.94,	493.88,	987.77,	1975.53, 3951.07, 7902.13], # B	    
]

def note_freq(note:int, octave:int=4 ) -> float:
    """
                OCTAVE 0	OCTAVE 1	OCTAVE 2	OCTAVE 3	OCTAVE 4	OCTAVE 5	OCTAVE 6	OCTAVE 7	OCTAVE 8
        C	    16.35 Hz	32.70 Hz	65.41 Hz	130.81 Hz	261.63 Hz	523.25 Hz	1046.50 Hz	2093.00 Hz	4186.01 Hz
        C#/Db	17.32 Hz	34.65 Hz	69.30 Hz	138.59 Hz	277.18 Hz	554.37 Hz	1108.73 Hz	2217.46 Hz	4434.92 Hz
        D	    18.35 Hz	36.71 Hz	73.42 Hz	146.83 Hz	293.66 Hz	587.33 Hz	1174.66 Hz	2349.32 Hz	4698.63 Hz
        D#/Eb	19.45 Hz	38.89 Hz	77.78 Hz	155.56 Hz	311.13 Hz	622.25 Hz	1244.51 Hz	2489.02 Hz	4978.03 Hz
        E	    20.60 Hz	41.20 Hz	82.41 Hz	164.81 Hz	329.63 Hz	659.25 Hz	1318.51 Hz	2637.02 Hz	5274.04 Hz
        F	    21.83 Hz	43.65 Hz	87.31 Hz	174.61 Hz	349.23 Hz	698.46 Hz	1396.91 Hz	2793.83 Hz	5587.65 Hz
        F#/Gb	23.12 Hz	46.25 Hz	92.50 Hz	185.00 Hz	369.99 Hz	739.99 Hz	1479.98 Hz	2959.96 Hz	5919.91 Hz
        G	    24.50 Hz	49.00 Hz	98.00 Hz	196.00 Hz	392.00 Hz	783.99 Hz	1567.98 Hz	3135.96 Hz	6271.93 Hz
        G#/Ab	25.96 Hz	51.91 Hz	103.83 Hz	207.65 Hz	415.30 Hz	830.61 Hz	1661.22 Hz	3322.44 Hz	6644.88 Hz
        A	    27.50 Hz	55.00 Hz	110.00 Hz	220.00 Hz	440.00 Hz	880.00 Hz	1760.00 Hz	3520.00 Hz	7040.00 Hz
        A#/Bb	29.14 Hz	58.27 Hz	116.54 Hz	233.08 Hz	466.16 Hz	932.33 Hz	1864.66 Hz	3729.31 Hz	7458.62 Hz
        B	    30.87 Hz	61.74 Hz	123.47 Hz	246.94 Hz	493.88 Hz	987.77 Hz	1975.53 Hz	3951.07 Hz	7902.13 Hz
    """
    # Check inputs:
    assert isinstance(octave, int)
    assert isinstance(note, int)
    assert 0 <= octave <= 8
    assert 0 <= note < 12
    # Choose note:
    return NOTES_TABLE[note][octave]


def _test():
    # ascend()
    alarm()

if __name__ == "__main__":
    _test()