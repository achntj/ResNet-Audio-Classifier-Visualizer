import Link from "next/link";

export default function About() {
  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            About the Audio Classifier
          </h1>

          <p className="text-stone-600">
            How this deep learning model understands environmental sounds
          </p>

          <Link
            href="/"
            className="mt-2 inline-block text-sm text-stone-500 hover:text-stone-700 hover:underline"
          >
            ← Back to classifier
          </Link>
        </div>

        <div className="space-y-8 text-stone-700">
          <section className="space-y-4">
            <h2 className="text-2xl font-light text-stone-800">How It Works</h2>
            <p>
              This system uses a specialized neural network architecture called
              a ResNet (Residual Network) to analyze environmental sounds. The
              model was trained on thousands of audio samples to recognize 50
              different categories of sounds.
            </p>
          </section>

          <section className="mt-8 space-y-4">
            <h2 className="text-2xl font-light text-stone-800">
              Model Architecture
            </h2>
            <div className="rounded-lg bg-white p-4 shadow-sm">
              <h3 className="mb-2 font-medium text-stone-800">
                ResNet-inspired CNN
              </h3>
              <ul className="list-disc space-y-1 pl-5 text-stone-700">
                <li>
                  <span className="font-medium">Input:</span> 128-band mel
                  spectrograms
                </li>
                <li>
                  <span className="font-medium">Layers:</span>
                  <ol className="mt-1 list-decimal space-y-1 pl-5">
                    <li>Initial conv (7x7, stride 2) + maxpool</li>
                    <li>4 residual blocks (64→128→256→512 channels)</li>
                    <li>Global average pooling + dropout</li>
                  </ol>
                </li>
                <li>
                  <span className="font-medium">Output:</span> 50-class
                  probabilities
                </li>
              </ul>
              <p className="mt-3 text-sm text-stone-500">
                Trained on ESC-50 dataset with label smoothing and mixup
                augmentation.
              </p>
            </div>
          </section>

          <section className="space-y-4">
            <h2 className="text-2xl font-light text-stone-800">
              Technical Stack
            </h2>
            <div className="rounded-lg bg-white p-4 shadow-sm">
              <ul className="list-disc space-y-1 pl-5 text-stone-700">
                <li>
                  <span className="font-medium">Model:</span> PyTorch CNN with
                  residual blocks
                </li>
                <li>
                  <span className="font-medium">Backend:</span> Modal for
                  serverless GPU inference
                </li>
                <li>
                  <span className="font-medium">Frontend:</span> Next.js with
                  TypeScript and Tailwind CSS
                </li>
                <li>
                  <span className="font-medium">Audio Processing:</span>{" "}
                  TorchAudio Mel spectrograms
                </li>
              </ul>
              <p className="mt-3 text-sm text-stone-500">
                The entire pipeline from audio upload to visualization runs in
                browser and cloud.
              </p>
            </div>
          </section>

          <section className="space-y-4">
            <h2 className="text-2xl font-light text-stone-800">
              Understanding the Visualizations
            </h2>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-2 font-medium text-stone-800">
                  Input Spectrogram
                </h3>
                <p>
                  A mel spectrogram showing frequency (vertical axis) over time
                  (horizontal axis). The color intensity represents amplitude
                  (louder sounds are brighter). This is the initial audio
                  representation the model processes.
                </p>
              </div>
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-2 font-medium text-stone-800">
                  Top Predictions
                </h3>
                <p>
                  The model{"'"}s top 3 classifications with confidence
                  percentages. Each prediction includes an emoji representing
                  the sound category. The highest confidence result is
                  highlighted with a primary badge.
                </p>
              </div>
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-2 font-medium text-stone-800">
                  Convolutional Layers
                </h3>
                <p>
                  Shows the main processing layers (conv1, layer1-layer4) and
                  their internal blocks. Each layer transforms the input, with
                  early layers detecting simple patterns and deeper layers
                  identifying complex sound structures.
                </p>
              </div>
              <div className="rounded-lg bg-white p-4 shadow-sm">
                <h3 className="mb-2 font-medium text-stone-800">
                  Audio Waveform
                </h3>
                <p>
                  A downsampled version of your original audio waveform showing
                  amplitude variations over time. The duration and sample rate
                  are displayed to provide context about the audio
                  characteristics.
                </p>
              </div>
            </div>
            <div className="mt-4 text-sm text-stone-500">
              <p>
                All visualizations include a color scale showing the value
                ranges from -1 to 1.
              </p>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
