// Form handling for login and signup
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const messageDiv = document.getElementById('message');
    const dobInput = document.getElementById('birthday');
    const ageInput = document.getElementById('age');
    const dobError = document.getElementById('dobError');

    // Set max to today's date; prevent future dates
    try {
        const today = new Date();
        const yyyy = today.getFullYear();
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const dd = String(today.getDate()).padStart(2, '0');
        const todayStr = `${yyyy}-${mm}-${dd}`;
        if (dobInput) {
            dobInput.setAttribute('max', todayStr);
        }
    } catch(_) {}

    function calculateAge(isoDate) {
        const dob = new Date(isoDate);
        const today = new Date();
        let age = today.getFullYear() - dob.getFullYear();
        const m = today.getMonth() - dob.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
            age--;
        }
        return age;
    }

    function isValidDob(isoDate) {
        if (!isoDate || !/^\d{4}-\d{2}-\d{2}$/.test(isoDate)) return false;
        const min = new Date('1950-01-01');
        const max = new Date();
        const d = new Date(isoDate);
        if (Number.isNaN(d.getTime())) return false;
        return d >= min && d <= max;
    }

    function showDobError(show) {
        if (!dobError) return;
        dobError.style.display = show ? 'block' : 'none';
    }

    if (dobInput) {
        // Prevent invalid manual typing (e.g., non-date or out-of-range)
        dobInput.addEventListener('input', function() {
            const val = this.value;
            const ok = isValidDob(val);
            showDobError(!ok);
            if (ok && ageInput) {
                const age = calculateAge(val);
                ageInput.value = age;
            } else if (ageInput) {
                ageInput.value = '';
            }
        });
        // On blur, enforce validity
        dobInput.addEventListener('blur', function() {
            const val = this.value;
            if (!isValidDob(val)) {
                showDobError(true);
                this.value = '';
                if (ageInput) ageInput.value = '';
            }
        });
        // Block non-digit and non-hyphen keystrokes to mitigate random years like 0678
        dobInput.addEventListener('keypress', function(e) {
            const allowed = /[0-9\-]/;
            if (!allowed.test(e.key)) {
                e.preventDefault();
            }
        });
    }

    // Login form handling
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            
            // Basic validation
            if (!email || !password) {
                showMessage('Please fill in all fields', 'error');
                return;
            }
            
            // Email validation
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                showMessage('Please enter a valid email address', 'error');
                return;
            }
            
            // Disable submit button
            const submitBtn = loginForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Signing in...';
            submitBtn.disabled = true;
            
            // Send login request
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: email,
                    password: password
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    showMessage(data.message, 'success');
                    // Use a more reliable redirect method
                    setTimeout(() => {
                        if (data.redirect) {
                            window.location.replace(data.redirect);
                        } else {
                            window.location.replace('/dashboard');
                        }
                    }, 1000);
                } else {
                    showMessage(data.message || 'Login failed', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    showMessage('Network error. Please check your connection and try again.', 'error');
                } else {
                    showMessage('An error occurred. Please try again.', 'error');
                }
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });
    }

    // Signup form handling
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const name = document.getElementById('name').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const birthday = document.getElementById('birthday').value;
            const age = document.getElementById('age').value;
            
            // Basic validation
            if (!name || !email || !password || !confirmPassword || !birthday || !age) {
                showMessage('Please fill in all fields', 'error');
                return;
            }

            // Frontend DOB validation window (1950-01-01..today)
            if (!isValidDob(birthday)) {
                showDobError(true);
                showMessage('Please enter a valid date of birth', 'error');
                return;
            }
            
            if (password !== confirmPassword) {
                showMessage('Passwords do not match', 'error');
                return;
            }
            
            if (password.length < 6) {
                showMessage('Password must be at least 6 characters long', 'error');
                return;
            }
            
            // Email validation
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                showMessage('Please enter a valid email address', 'error');
                return;
            }
            
            // Disable submit button
            const submitBtn = signupForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Creating account...';
            submitBtn.disabled = true;
            
            // Send signup request
            fetch('/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: name,
                    email: email,
                    password: password,
                    birthday: birthday,
                    age: age
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    showMessage(data.message, 'success');
                    // Use a more reliable redirect method
                    setTimeout(() => {
                        if (data.redirect) {
                            window.location.replace(data.redirect);
                        } else {
                            window.location.replace('/dashboard');
                        }
                    }, 1000);
                } else {
                    showMessage(data.message || 'Registration failed', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    showMessage('Network error. Please check your connection and try again.', 'error');
                } else {
                    showMessage('An error occurred. Please try again.', 'error');
                }
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });
    }

    // Function to show messages
    function showMessage(message, type) {
        if (messageDiv) {
            messageDiv.textContent = message;
            messageDiv.className = `message ${type}`;
            messageDiv.style.display = 'block';
            
            // Auto-hide success messages after 5 seconds
            if (type === 'success') {
                setTimeout(() => {
                    messageDiv.textContent = '';
                    messageDiv.className = 'message';
                    messageDiv.style.display = 'none';
                }, 5000);
            }
        }
    }

    // Real-time password confirmation validation
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const passwordInput = document.getElementById('password');
    
    if (confirmPasswordInput && passwordInput) {
        confirmPasswordInput.addEventListener('input', function() {
            const password = passwordInput.value;
            const confirmPassword = this.value;
            
            if (confirmPassword && password !== confirmPassword) {
                this.style.borderColor = '#dc3545';
            } else {
                this.style.borderColor = '#e1e5e9';
            }
        });
        
        passwordInput.addEventListener('input', function() {
            const password = this.value;
            const confirmPassword = confirmPasswordInput.value;
            
            if (confirmPassword && password !== confirmPassword) {
                confirmPasswordInput.style.borderColor = '#dc3545';
            } else {
                confirmPasswordInput.style.borderColor = '#e1e5e9';
            }
        });
    }

    // Add loading states to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.type === 'submit') {
                // Loading state is handled in form submission
                return;
            }
            
            // Add loading state for other buttons
            const originalText = this.textContent;
            this.textContent = 'Loading...';
            this.disabled = true;
            
            setTimeout(() => {
                this.textContent = originalText;
                this.disabled = false;
            }, 2000);
        });
    });
}); 