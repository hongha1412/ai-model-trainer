<template>
  <div class="app-container">
    <header class="app-header">
      <div class="logo-container">
        <img src="/favicon.svg" alt="AI Model Server Logo" class="logo" />
        <h1 class="app-title">AI Model Server</h1>
      </div>
      <nav class="app-nav">
        <router-link to="/" class="nav-link">Dashboard</router-link>
        <router-link to="/model-config" class="nav-link">Model Config</router-link>
        <router-link to="/model-training" class="nav-link">Model Training</router-link>
        <router-link to="/openapi" class="nav-link">API Docs</router-link>
      </nav>
    </header>
    
    <main class="app-content">
      <router-view />
    </main>
    
    <footer class="app-footer">
      <p>&copy; {{ new Date().getFullYear() }} AI Model Server</p>
    </footer>
  </div>
</template>

<style>
:root {
  --color-primary: #10A37F;
  --color-primary-dark: #0E8C6D;
  --color-secondary: #202123;
  --color-background: #FFFFFF;
  --color-text: #353740;
  --color-text-light: #6E6E80;
  --color-border: #E5E5E5;
  --color-error: #D92F2F;
  --color-success: #10A37F;
  --border-radius: 6px;
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
}

body {
  font-family: 'Inter', sans-serif;
  color: var(--color-text);
  margin: 0;
  padding: 0;
  background-color: var(--color-background);
  line-height: 1.5;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background-color: var(--color-background);
  border-bottom: 1px solid var(--color-border);
  padding: var(--spacing-md) var(--spacing-xl);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo-container {
  display: flex;
  align-items: center;
}

.logo {
  width: 32px;
  height: 32px;
  margin-right: var(--spacing-md);
}

.app-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
  color: var(--color-primary);
}

.app-nav {
  display: flex;
  gap: var(--spacing-lg);
}

.nav-link {
  color: var(--color-text);
  text-decoration: none;
  font-weight: 500;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius);
  transition: all 0.2s ease;
}

.nav-link:hover {
  color: var(--color-primary);
}

.router-link-active {
  color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.1);
}

.app-content {
  flex: 1;
  padding: var(--spacing-xl);
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.app-footer {
  border-top: 1px solid var(--color-border);
  padding: var(--spacing-md) var(--spacing-xl);
  text-align: center;
  color: var(--color-text-light);
  font-size: 0.875rem;
}

/* Common components */
.card {
  background-color: var(--color-background);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.card-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  color: var(--color-secondary);
}

.btn {
  display: inline-block;
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius);
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: all 0.2s ease;
  text-decoration: none;
}

.btn-primary {
  background-color: var(--color-primary);
  color: white;
}

.btn-primary:hover {
  background-color: var(--color-primary-dark);
}

.btn-secondary {
  background-color: transparent;
  color: var(--color-text);
  border: 1px solid var(--color-border);
}

.btn-secondary:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.btn-danger {
  background-color: var(--color-error);
  color: white;
}

.btn-danger:hover {
  background-color: #B82727;
}

.form-group {
  margin-bottom: var(--spacing-md);
}

.form-label {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 500;
}

.form-control {
  width: 100%;
  padding: var(--spacing-sm);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  font-family: 'Inter', sans-serif;
  font-size: 1rem;
}

.form-control:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
}

.alert {
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
  margin-bottom: var(--spacing-md);
}

.alert-success {
  background-color: rgba(16, 163, 127, 0.1);
  color: var(--color-success);
  border: 1px solid rgba(16, 163, 127, 0.2);
}

.alert-error {
  background-color: rgba(217, 47, 47, 0.1);
  color: var(--color-error);
  border: 1px solid rgba(217, 47, 47, 0.2);
}

.code-block {
  font-family: 'Roboto Mono', monospace;
  background-color: var(--color-secondary);
  color: white;
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
  overflow-x: auto;
}

.loading-spinner {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(16, 163, 127, 0.3);
  border-radius: 50%;
  border-top-color: var(--color-primary);
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.badge-success {
  background-color: rgba(16, 163, 127, 0.1);
  color: var(--color-success);
}

.badge-warning {
  background-color: rgba(255, 170, 0, 0.1);
  color: #FFAA00;
}

.table {
  width: 100%;
  border-collapse: collapse;
}

.table th, .table td {
  padding: var(--spacing-sm);
  text-align: left;
  border-bottom: 1px solid var(--color-border);
}

.table th {
  font-weight: 600;
  color: var(--color-text-light);
}

.table tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.02);
}

@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
  }
  
  .app-nav {
    width: 100%;
    justify-content: space-between;
  }
  
  .app-content {
    padding: var(--spacing-md);
  }
}
</style>