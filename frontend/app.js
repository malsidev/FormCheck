const exerciseSelect = document.getElementById('exercise');
const form = document.getElementById('analysis-form');
const fileInput = document.getElementById('video-file');
const resultCard = document.getElementById('result-card');
const scoreValue = document.getElementById('score-value');
const statusText = document.getElementById('status-text');
const messageText = document.getElementById('message-text');
const detailStatus = document.getElementById('detail-status');
const detailFile = document.getElementById('detail-file');
const detailExercise = document.getElementById('detail-exercise');

async function loadExercises() {
  const response = await fetch('/api/exercises');
  const data = await response.json();

  exerciseSelect.innerHTML = '';
  data.exercises.forEach((exercise) => {
    const option = document.createElement('option');
    option.value = exercise;
    option.textContent = exercise === 'squat' ? 'Присед' : 'Отжимание';
    exerciseSelect.appendChild(option);
  });
}

function updateResultCard(status, score, message) {
  resultCard.className = `result-card ${status}`;
  scoreValue.textContent = `${score ?? 0}%`;
  statusText.textContent = status === 'good' ? 'Техника в норме' : status === 'warning' ? 'Есть замечания по технике' : 'Техника требует корректировки';
  messageText.textContent = message || 'Результат готов.';
  detailStatus.textContent = status;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    alert('Загрузите фото или видео');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('exercise', exerciseSelect.value);

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    updateResultCard(result.status, result.score, result.message);
    detailFile.textContent = result.filename || file.name;
    detailExercise.textContent = result.exercise || exerciseSelect.value;
  } catch (error) {
    updateResultCard('bad', 0, 'Не удалось выполнить анализ. Проверьте файл и попробуйте ещё раз.');
    detailFile.textContent = file.name;
    detailExercise.textContent = exerciseSelect.value;
    console.error(error);
  }
});

loadExercises();
