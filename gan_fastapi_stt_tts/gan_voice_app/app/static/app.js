// FastAPI가 제공하는 각 API 주소를 정의한다.
const API = { stt: '/api/stt', generate: '/api/generate', tts: '/api/tts' };
// HTML에서 필요한 입력창과 버튼을 가져온다.
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const generateBtn = document.getElementById('generateBtn');
const digitSelect = document.getElementById('digit');
const countSelect = document.getElementById('count');
const statusText = document.getElementById('status');
const transcriptText = document.getElementById('transcript');
const generatedImage = document.getElementById('generatedImage');
const downloadLink = document.getElementById('downloadLink');
const ttsAudio = document.getElementById('ttsAudio');
// 녹음 객체와 미디어 스트림, 브라우저 Blob URL을 보관한다.
let recorder = null;
let mediaStream = null;
let imageUrl = null;
let audioUrl = null;
let busy = false;

// 서버가 반환한 오류 응답을 사람이 읽을 수 있는 문장으로 변환한다.
async function checkResponse(response) {
  if (response.ok) return response;
  const body = await response.json().catch(() => ({}));
  throw new Error(body.detail || `HTTP ${response.status}`);
}

// 버튼 활성 상태를 갱신하여 중복 요청을 방지한다.
function setBusy(value) {
  busy = value;
  generateBtn.disabled = value;
  recordBtn.disabled = value;
}

// 숫자 조건을 CGAN 서버에 전달하여 이미지를 생성하고 TTS를 요청한다.
async function generateDigit(digit) {
  setBusy(true);
  const count = Number(countSelect.value);
  try {
    statusText.textContent = `숫자 ${digit} 이미지 생성 중...`;
    // CGAN이 생성한 PNG 이미지를 Blob 형태로 수신한다.
    const response = await fetch(API.generate, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ digit, count })
    });
    await checkResponse(response);
    const blob = await response.blob();
    // 이전에 생성한 Blob URL을 해제하여 브라우저 메모리 누수를 줄인다.
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(blob);
    generatedImage.src = imageUrl;
    generatedImage.hidden = false;
    downloadLink.href = imageUrl;
    downloadLink.download = `digit_${digit}_${count}.png`;
    downloadLink.hidden = false;
    statusText.textContent = `숫자 ${digit} 이미지 ${count}장 생성 완료`;
    // 이미지 생성은 성공했으므로 TTS 오류는 별도로 안내한다.
    try {
      const speechResponse = await fetch(API.tts, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ digit, count })
      });
      await checkResponse(speechResponse);
      const audioBlob = await speechResponse.blob();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      audioUrl = URL.createObjectURL(audioBlob);
      ttsAudio.src = audioUrl;
      // 자동 재생이 차단되면 사용자가 오디오 컨트롤에서 직접 재생할 수 있다.
      await ttsAudio.play().catch(() => {
        statusText.textContent += ' (음성은 재생 버튼을 눌러 주세요)';
      });
    } catch (error) {
      statusText.textContent += ` (TTS 오류: ${error.message})`;
    }
  } catch (error) {
    statusText.textContent = `오류: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

// 녹음 시작 버튼을 누르면 브라우저의 마이크 접근 권한을 요청한다.
recordBtn.addEventListener('click', async () => {
  if (busy) return;
  try {
    // 마이크 사용은 localhost 또는 HTTPS 환경에서 허용된다.
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // 브라우저에서 지원하는 오디오 형식을 우선순위에 따라 선택한다.
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    const mimeType = candidates.find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error('이 브라우저에서 지원되는 녹음 형식이 없습니다.');
    recorder = new MediaRecorder(mediaStream, { mimeType });
    const chunks = [];
    recorder.ondataavailable = event => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    // 녹음이 종료되면 음성 파일을 STT API로 전송한다.
    recorder.onstop = async () => {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
      recordBtn.disabled = true;
      stopBtn.disabled = true;
      setBusy(true);
      try {
        const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
        const audioBlob = new Blob(chunks, { type: mimeType });
        const form = new FormData();
        form.append('audio', audioBlob, `speech.${extension}`);
        statusText.textContent = '음성을 텍스트로 변환 중...';
        const response = await fetch(API.stt, { method: 'POST', body: form });
        await checkResponse(response);
        const result = await response.json();
        transcriptText.textContent = result.transcript;
        digitSelect.value = String(result.digit);
        setBusy(false);
        // 인식한 숫자를 그대로 사용하여 이미지 생성과 음성 안내를 수행한다.
        await generateDigit(result.digit);
      } catch (error) {
        statusText.textContent = `오류: ${error.message}`;
      } finally {
        setBusy(false);
        stopBtn.disabled = true;
      }
    };
    recorder.start();
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    statusText.textContent = '녹음 중... 말씀하신 후 종료 버튼을 누르세요.';
  } catch (error) {
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
    statusText.textContent = `마이크 오류: ${error.message}`;
  }
});

// 녹음 종료 버튼을 누르면 onstop 이벤트가 STT 요청을 수행한다.
stopBtn.addEventListener('click', () => {
  if (recorder && recorder.state === 'recording') recorder.stop();
});

// 마이크 없이도 숫자 선택창으로 CGAN 이미지 생성 기능을 시험할 수 있다.
generateBtn.addEventListener('click', () => generateDigit(Number(digitSelect.value)));
