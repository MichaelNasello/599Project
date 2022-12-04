from utils import augment_patient
import matplotlib.pyplot as plt
from dataloader import load_patient_task_data_from_txt,clean_and_verify

def main():
    original = load_patient_task_data_from_txt('001',1)
    original = clean_and_verify(original)
    augmented = augment_patient(original)

    print("Now displaying the original and augmented signal for the patient")
    print("Noise is small and hard to see. Zoom in to see better. I recommend zooming in until u can see the dashed lines")

    assert original.shape == augmented.shape
    for signal in original.columns[1:]:
        assert original[signal].shape == augmented[signal].shape
        plt.plot(original[signal],'r--')
        plt.plot(augmented[signal],'b--')
        plt.title(signal)
        plt.legend(['original','augmented'])
        plt.show()


if __name__ == '__main__':
    main()
