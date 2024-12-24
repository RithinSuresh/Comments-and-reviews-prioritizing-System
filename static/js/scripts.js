window.onload = function() {
    const form = document.querySelector('form');
    const processingText = document.createElement('p');
    processingText.id = 'processing-text';
    processingText.textContent = 'Processing... Please wait.';
    processingText.style.color = '#0073e6';
    processingText.style.textAlign = 'center';
    processingText.style.display = 'none';  // Hidden by default

    form.appendChild(processingText);

    form.onsubmit = function() {
        const button = form.querySelector('button');
        button.disabled = true;
        button.textContent = 'Processing...';
        processingText.style.display = 'block';  // Show processing text
    };
};
