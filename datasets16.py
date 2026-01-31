from moabb.datasets import PhysionetMI, Weibo2014

class PhysionetMI16(PhysionetMI):
    """
    Physionet Motor Imagery dataset with 16 selected channels.
    
    The channels are motor-cortex focused and include:
    ['FCz', 'Pz', 'FC2', 'Cz', 'FC4', 'C3', 'CP1', 'C2', 'CP4', 'CP3', 'CP2', 'C4', 'CPz', 'FC3', 'C1', 'FC1']
    """

    def __init__(self, imagined=True, executed=False):
        super().__init__(imagined=imagined, executed=executed)
        self.code = "PhysionetMotorImagery16"
        self.selected_channels = [
            'FCz', 'Pz', 'FC2', 'Cz', 'FC4', 'C3', 'CP1', 'C2', 
            'CP4', 'CP3', 'CP2', 'C4', 'CPz', 'FC3', 'C1', 'FC1'
        ]

    def _load_one_run(self, subject, run, preload=True):
        # Load the original data (64 ch)
        raw = super()._load_one_run(subject, run, preload=preload)
        # Pick only the selected 16 channels
        raw.pick_channels(self.selected_channels)
        
        return raw

class Weibo2014_16(Weibo2014):
    """
    Weibo2014 dataset with 16 selected channels.
    
    The channels are motor-cortex focused and include:
    ['FCz', 'Pz', 'FC2', 'Cz', 'FC4', 'C3', 'CP1', 'C2', 'CP4', 'CP3', 'CP2', 'C4', 'CPz', 'FC3', 'C1', 'FC1']
    """

    def __init__(self):
        super().__init__()
        self.code = "Weibo2014_16"
        self.selected_channels = [
            'FCz', 'Pz', 'FC2', 'Cz', 'FC4', 'C3', 'CP1', 'C2', 
            'CP4', 'CP3', 'CP2', 'C4', 'CPz', 'FC3', 'C1', 'FC1'
        ]
        self.event_id = dict(
                left_hand=1,
                right_hand=2,
                hands=3,
                feet=4,
                # left_hand_right_foot=5,
                # right_hand_left_foot=6,
                rest=7,
            )
        print(type(self.event_id))

    def _load_one_run(self, subject, run, preload=True):
        # Load the original data (64 ch)
        raw = super()._load_one_run(subject, run, preload=preload)
        # Pick only the selected 16 channels
        raw.pick_channels(self.selected_channels)
        
        return raw