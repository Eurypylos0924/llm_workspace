const form = document.querySelector('#tts-form');
const speed = document.querySelector('#speed');
const speedValue = document.querySelector('#speed-value');
const statusText = document.querySelector('#status');
const result = document.querySelector('#result');
const audio = document.querySelector('#audio');
const download = document.querySelector('#download');
const submit = document.querySelector('#submit');
let audioUrl = null;
speed.addEventListener('input', () => {
  speedValue.textContent = Number(speed.value).toFixed(2);
});
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true;
  result.hidden = true;
  statusText.textContent = '음성을 생성하고 있습니다...';
  try {
    const response = await fetch('/api/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text: document.querySelector('#text').value,
        voice: document.querySelector('#voice').value,
        speed: Number(speed.value),
      }),
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        message = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      } catch (_) {}
      throw new Error(message);
    }
    const blob = await response.blob();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(blob);
    audio.src = audioUrl;
    download.href = audioUrl;
    result.hidden = false;
    statusText.textContent = '음성 생성 완료';
  } catch (error) {
    statusText.textContent = `오류: ${error.message}`;
  } finally {
    submit.disabled = false;
  }
});
