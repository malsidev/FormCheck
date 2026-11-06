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
const feedbackList = document.getElementById('feedback-list');
const uploadedPreview = document.getElementById('uploaded-preview');
const uploadedVideo = document.getElementById('uploaded-video');

function getExerciseLabel(exercise) {
  const labels = {
    auto: 'Определить автоматически',
    squat: 'Присед',
    'push-up': 'Отжимание',
    pushup: 'Отжимание',
    plank: 'Планка',
    lunge: 'Выпад',
    lunges: 'Выпад',
    deadlift: 'Тяга',
  };

  return labels[exercise] || exercise;
}

async function loadExercises() {
  const response = await fetch('/api/exercises');
  const data = await response.json();

  exerciseSelect.innerHTML = '';
  data.exercises.forEach((exercise) => {
    const option = document.createElement('option');
    option.value = exercise;
    option.textContent = getExerciseLabel(exercise);
    if (exercise === 'auto') {
      option.selected = true;
    }
    exerciseSelect.appendChild(option);
  });
}

function showUploadedPreview(file) {
  uploadedPreview.src = '';
  uploadedPreview.style.display = 'none';
  uploadedVideo.src = '';
  uploadedVideo.style.display = 'none';
  uploadedVideo.removeAttribute('src');

  if (!file) {
    return;
  }

  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (event) => {
      uploadedPreview.src = event.target.result;
      uploadedPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    return;
  }

  if (file.type.startsWith('video/')) {
    uploadedVideo.src = URL.createObjectURL(file);
    uploadedVideo.style.display = 'block';
    uploadedVideo.load();
  }
}

function updateResultCard(status, score, message, advice = [], issues = []) {
  resultCard.className = `result-card ${status}`;
  scoreValue.textContent = `${score ?? 0}%`;

  const normalizedMessage = message || 'Результат готов.';
  const recommendations = Array.isArray(advice) && advice.length ? advice : (Array.isArray(issues) ? issues : []);

  if (status === 'good') {
    statusText.textContent = 'Техника в норме';
    messageText.textContent = normalizedMessage === 'Техника приседа в норме' || normalizedMessage === 'Техника отжиманий нормальная' || normalizedMessage === 'Планка выполнена ровно' || normalizedMessage === 'Выпад выполнен уверенно' || normalizedMessage === 'Техника тяги в норме' ? normalizedMessage : 'Техника упражнения выполнена в корректной форме.';
  } else if (status === 'warning') {
    statusText.textContent = 'Есть замечания по технике';
    messageText.textContent = normalizedMessage || 'Есть небольшие отклонения от правильной техники.';
  } else {
    statusText.textContent = 'Техника требует корректировки';
    messageText.textContent = normalizedMessage || 'Нужно скорректировать технику упражнения.';
  }

  detailStatus.textContent = status;

  if (recommendations.length) {
    feedbackList.innerHTML = recommendations.map((item) => `<li>${item}</li>`).join('');
  } else {
    feedbackList.innerHTML = '<li>Замечаний не найдено. Сохраняйте текущую форму.</li>';
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    alert('Загрузите фото или видео');
    return;
  }

  showUploadedPreview(file);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('exercise', exerciseSelect.value || 'auto');

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    updateResultCard(result.status, result.score, result.message, result.advice || [], result.issues || []);
    detailFile.textContent = result.filename || file.name;
    detailExercise.textContent = getExerciseLabel(result.exercise || exerciseSelect.value);

    if (result.uploaded_image) {
      uploadedPreview.src = result.uploaded_image;
      uploadedPreview.style.display = 'block';
    } else if (file.type.startsWith('video/')) {
      uploadedVideo.src = URL.createObjectURL(file);
      uploadedVideo.style.display = 'block';
      uploadedVideo.load();
    }
  } catch (error) {
    updateResultCard('bad', 0, 'Не удалось выполнить анализ. Проверьте файл и попробуйте ещё раз.', ['Проверьте файл и повторите загрузку.']);
    detailFile.textContent = file.name;
    detailExercise.textContent = getExerciseLabel(exerciseSelect.value);
    console.error(error);
  }
});

loadExercises();
