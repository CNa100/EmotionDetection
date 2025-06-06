const imgEl = document.getElementById('inputImage');
const videoEl = document.getElementById('video');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');

let session;
ort.InferenceSession.create("best.onnx").then(s => session = s);

function handleFileUpload(event) {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = () => {
    imgEl.src = reader.result;
    imgEl.classList.remove('hidden');
    videoEl.classList.add('hidden');
    canvas.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

function openCamera(useFrontCamera) {
  navigator.mediaDevices.getUserMedia({
    video: { facingMode: useFrontCamera ? "user" : "environment" }
  })
  .then(stream => {
    videoEl.srcObject = stream;
    videoEl.classList.remove('hidden');
    imgEl.classList.add('hidden');
    canvas.classList.add('hidden');
  })
  .catch(err => alert("Camera access denied."));
}

async function detect() {
  canvas.width = 640;
  canvas.height = 640;
  canvas.classList.remove("hidden");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 640;
  tempCanvas.height = 640;
  const tempCtx = tempCanvas.getContext("2d");

  if (!imgEl.classList.contains("hidden")) {
    tempCtx.drawImage(imgEl, 0, 0, 640, 640);
  } else if (!videoEl.classList.contains("hidden")) {
    tempCtx.drawImage(videoEl, 0, 0, 640, 640);
  } else {
    alert("No input available.");
    return;
  }

  const imageData = tempCtx.getImageData(0, 0, 640, 640);
  const input = new ort.Tensor("float32", new Float32Array(3 * 640 * 640), [1, 3, 640, 640]);

  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j++) {
    input.data[j] = imageData.data[i] / 255;
    input.data[j + 640 * 640] = imageData.data[i + 1] / 255;
    input.data[j + 2 * 640 * 640] = imageData.data[i + 2] / 255;
  }

  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  const classNames = [
    "Angry", "Disgust", "Excited", "Fear", "Happy",
    "Sad", "Serious", "Thinking", "Worried", "neutral"
  ];

  const numDetections = output.length / 15;
  console.log("🔎 Total boxes:", numDetections);

  ctx.drawImage(tempCanvas, 0, 0);

  let anyDetected = false;

  for (let i = 0; i < numDetections; i++) {
    const offset = i * 15;
    const x = output[offset];
    const y = output[offset + 1];
    const w = output[offset + 2];
    const h = output[offset + 3];
    const objConf = output[offset + 4];
    const classScores = output.slice(offset + 5, offset + 15);
    const maxScore = Math.max(...classScores);
    const classIndex = classScores.indexOf(maxScore);
    const conf = objConf * maxScore;

    if (conf > 0.2) {
      anyDetected = true;
      const left = (x - w / 2);
      const top = (y - h / 2);

      console.log(`✅ Detection: ${classNames[classIndex]} (${(conf * 100).toFixed(1)}%)`);

      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(left, top, w, h);

      ctx.fillStyle = "lime";
      ctx.font = "16px Arial";
      ctx.fillText(`${classNames[classIndex]} (${(conf * 100).toFixed(1)}%)`, left, top - 5);
    }
  }

  if (!anyDetected) {
    console.log("⚠️ هیچ چیزی تشخیص داده نشد.");
  }
}
