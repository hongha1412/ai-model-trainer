// Initialize Feather icons
document.addEventListener('DOMContentLoaded', function() {
  feather.replace();

  // Add automatic fading for alert messages
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(function(alert) {
    setTimeout(function() {
      // Get the Bootstrap alert instance and close it
      const bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    }, 5000); // Close after 5 seconds
  });

  // Handle sidebar toggle for mobile
  const toggleSidebarBtn = document.getElementById('sidebarToggle');
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', function() {
      document.getElementById('sidebar').classList.toggle('show');
    });
  }

  // Add confirmation for model unloading
  const unloadForms = document.querySelectorAll('form[action*="unload"]');
  unloadForms.forEach(function(form) {
    form.addEventListener('submit', function(e) {
      if (!confirm('Are you sure you want to unload this model?')) {
        e.preventDefault();
      }
    });
  });

  // Add tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function(tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
});

// Helper function to format JSON
function formatJSON(json) {
  if (typeof json === 'string') {
    try {
      json = JSON.parse(json);
    } catch (e) {
      return json;
    }
  }
  return JSON.stringify(json, null, 2);
}

// Helper function to show API endpoint examples
function showApiExample(endpoint, model) {
  const baseUrl = window.location.origin;
  const apiUrl = `${baseUrl}/api/v1/${endpoint}`;
  
  let requestBody = {};
  let exampleCode = '';
  
  if (endpoint === 'completions') {
    requestBody = {
      model: model || "your-model-id",
      prompt: "Hello, how are you?",
      max_tokens: 100,
      temperature: 0.7
    };
    
    exampleCode = `import requests

response = requests.post(
    "${apiUrl}",
    json=${formatJSON(requestBody)}
)

print(response.json())`;
  }
  else if (endpoint === 'chat/completions') {
    requestBody = {
      model: model || "your-model-id",
      messages: [
        {"role": "user", "content": "Hello, how are you?"}
      ],
      max_tokens: 100,
      temperature: 0.7
    };
    
    exampleCode = `import requests

response = requests.post(
    "${apiUrl}",
    json=${formatJSON(requestBody)}
)

print(response.json())`;
  }
  
  // Show in modal or other UI element
  if (document.getElementById('exampleModalBody')) {
    document.getElementById('exampleModalTitle').textContent = `API Example: ${endpoint}`;
    document.getElementById('exampleApiUrl').textContent = apiUrl;
    document.getElementById('exampleRequestBody').textContent = formatJSON(requestBody);
    document.getElementById('exampleCode').textContent = exampleCode;
    
    const exampleModal = new bootstrap.Modal(document.getElementById('exampleModal'));
    exampleModal.show();
  }
}
