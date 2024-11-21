from flask import Flask, render_template, request, url_for
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import io

app = Flask(__name__)

# Load all models into a dictionary
models = {
    "A": tf.keras.models.load_model("models/A.h5"),
    "E": tf.keras.models.load_model("models/E.h5"),
    "I": tf.keras.models.load_model("models/I.h5"),
    "O": tf.keras.models.load_model("models/O.h5"),
    "U": tf.keras.models.load_model("models/U.h5"),
    "ka-ka-ka": tf.keras.models.load_model("models/ka-ka-ka.h5"),
    "pakata": tf.keras.models.load_model("models/pakata.h5"),
    "pa-pa-pa": tf.keras.models.load_model("models/pa-pa-pa.h5"),
    "pataka": tf.keras.models.load_model("models/pataka.h5"),
    "petaka": tf.keras.models.load_model("models/petaka.h5"),
    "ta-ta-ta": tf.keras.models.load_model("models/ta-ta-ta.h5")
}

# Converting waveforms to spectograms
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    stfts = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(stfts)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

@app.route("/")
def home():
    vowel_audio_files = {
        "A": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_A.wav'),
        "E": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_E.wav'),
        "I": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_I.wav'),
        "O": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_O.wav'),
        "U": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_U.wav'),
    }
    ddk_audio_files = {
        "ka-ka-ka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_ka-ka-ka.wav'),
        "pakata": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pakata.wav'),
        "pa-pa-pa": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pa-pa-pa.wav'),
        "pataka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pataka.wav'),
        "petaka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_petaka.wav'),
        "ta-ta-ta": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_ta-ta-ta.wav'),
    }
    return render_template('index.html', vowel_audio_files=vowel_audio_files, ddk_audio_files=ddk_audio_files)

@app.route("/predict", methods=["POST"])
def predict():
    vowel_audio_files = {
        "A": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_A.wav'),
        "E": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_E.wav'),
        "I": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_I.wav'),
        "O": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_O.wav'),
        "U": url_for('static', filename='audio/examples/vowels/AVPEPUDEAC0001_U.wav'),
    }
    ddk_audio_files = {
        "ka-ka-ka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_ka-ka-ka.wav'),
        "pakata": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pakata.wav'),
        "pa-pa-pa": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pa-pa-pa.wav'),
        "pataka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_pataka.wav'),
        "petaka": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_petaka.wav'),
        "ta-ta-ta": url_for('static', filename='audio/examples/ddk/AVPEPUDEAC0001_ta-ta-ta.wav'),
    }

    files_to_delete = ['tmp/voice_message.wav', 'tmp/voice_message_processed.wav']

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

    # file = request.form.get('content')
    file = request.files['content']
    selected_option = request.form.get('selectedOption')  # Get the selected option from the form
    filename = secure_filename(file.filename)  # make sure the filename is secure
    filepath = os.path.join('tmp', filename)  # create a path to save the file
    file.save(filepath)  # save the file
    # file = tf.io.read_file(str(file))

    # Convert audio to 16-bit WAV, 16 kHz, mono using pydub
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # Ensure 16 kHz, mono, 16-bit
    wav_path = os.path.splitext(filepath)[0] + '_processed.wav'
    audio.export(wav_path, format='wav')
    filepath = wav_path

    file_bytes = file.read()
    file_tensor, sample_rate = tf.audio.decode_wav(tf.io.read_file(filepath), desired_channels=1, desired_samples=16000,)
    file_tensor = tf.squeeze(file_tensor, axis=-1)
    waveform = file_tensor
    file_tensor = get_spectrogram(file_tensor)
    file_tensor = tf.image.resize(file_tensor, [124, 129])
    file_tensor = file_tensor[tf.newaxis, ...]

    # Use the appropriate model based on the selected option
    model = models.get(selected_option)
    prediction = model.predict(file_tensor)
    print(prediction.shape)
    prediction_value = 'HC' if prediction[0][0] > prediction[0][1] else 'PD'
    return render_template("index.html", prediction=prediction_value, file=file_tensor, vowel_audio_files=vowel_audio_files, ddk_audio_files=ddk_audio_files)

if __name__ == "__main__":
    context = ('cert.pem', 'key.pem')  # Path to your certificate and key files
    app.run(debug=True, host='0.0.0.0', ssl_context=context)