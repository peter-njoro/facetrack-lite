document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector('form');
    if (!form) return;

    // Create loading overlay with progress bar
    const loader = document.createElement('div');
    loader.id = 'enroll-loading-overlay';
    loader.style.display = 'none';
    loader.style.position = 'fixed';
    loader.style.top = 0;
    loader.style.left = 0;
    loader.style.width = '100vw';
    loader.style.height = '100vh';
    loader.style.background = 'rgba(255, 255, 255, 0.8)';
    loader.style.zIndex = 9999;
    loader.style.justifyContent = 'center';
    loader.style.alignItems = 'center';
    loader.innerHTML = `
        <div style="text-align:center; min-width:300px;">
            <div style="margin-bottom:1rem;font-size:1.2rem;">Processing images, please wait...</div>
            <div class="progress" style="height: 2rem;">
                <div id="enroll-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-primary"
                    role="progressbar" style="width: 0%; font-size:1.1rem;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                    0%
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(loader);

    let pollInterval = null;

    form.addEventListener('submit', function () {
        loader.style.display = 'flex';

        // Start polling progress
        pollInterval = setInterval(function () {
            fetch('/enroll/progress/')
                .then(response => response.json())
                .then(data => {
                    const progressBar = document.getElementById('enroll-progress-bar');
                    let progress = data.progress || 0;
                    progressBar.style.width = progress + '%';
                    progressBar.innerText = progress + '%';
                    progressBar.setAttribute('aria-valuenow', progress);
                    if (progress >= 100) {
                        clearInterval(pollInterval);
                    }
                });
        }, 400);
    });
});