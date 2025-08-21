function fetchProgress() {
    const url = window.RECOGNITION_PROGRESS_URL;
    fetch(url)
      .then(response => response.json())
      .then(data => {
        document.getElementById('present-count').innerText = data.present_count;
        document.getElementById('total-expected').innerText = data.total_expected;
        document.getElementById('unknown-count').innerText = data.unknown_count;

        let percent = data.total_expected > 0
          ? Math.round((data.present_count / data.total_expected) * 100)
          : 0;

        let bar = document.getElementById('progress-bar');
        bar.style.width = percent + '%';
        bar.setAttribute('aria-valuenow', percent);
        bar.innerText = percent + '%';
      })
      .catch(error => console.error('Error fetching progress:', error));
}

if (window.RECOGNITION_PROGRESS_URL) {
    fetchProgress();
    setInterval(fetchProgress, 3000); 
}