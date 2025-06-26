    document.addEventListener("DOMContentLoaded", function() {

        // --- HIỆU ỨNG HEADER THU NHỎ KHI CUỘN ---
        const mainHeader = document.querySelector('.main-header');
        if (mainHeader) {
            window.addEventListener('scroll', () => {
                if (window.scrollY > 10) {
                    mainHeader.classList.add('scrolled');
                } else {
                    mainHeader.classList.remove('scrolled');
                }
            });
        }

        // --- HIỆU ỨNG ĐẾM SỐ ---
        function animateCountUp(el) {
            const targetValue = el.dataset.target;
            if (!targetValue) return;
            const target = parseFloat(targetValue.replace(/[^0-9.]/g, ''));
            if (isNaN(target)) return;

            const textSuffix = targetValue.replace(/[0-9.]/g, ''); 
            let count = 0;
            const duration = 2000;

            const updateCount = () => {
                const increment = target / (duration / 16); // ~60fps
                count += increment;

                if (count < target) {
                    let displayValue = targetValue.includes('.') ? count.toFixed(1) : Math.floor(count).toLocaleString('en-US');
                    el.innerText = displayValue + textSuffix;
                    requestAnimationFrame(updateCount);
                } else {
                    el.innerText = targetValue;
                }
            };
            requestAnimationFrame(updateCount);
        }

        // --- HIỆU ỨNG 3D TILT ĐỘNG THEO CHUỘT ---
        const tiltElements = document.querySelectorAll('.phone-mockup, .video-wrapper-longform');
        tiltElements.forEach(el => {
            const maxRotation = 8; // Độ nghiêng tối đa (độ)

            el.addEventListener('mousemove', (e) => {
                const rect = el.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const { width, height } = rect;

                const rotateY = maxRotation * ((x - width / 2) / (width / 2));
                const rotateX = -maxRotation * ((y - height / 2) / (height / 2));

                el.style.transform = `translateY(-10px) scale(1.03) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
                el.style.boxShadow = '0 30px 40px -15px rgba(0,0,0,0.3)';
            });

            el.addEventListener('mouseleave', () => {
                el.style.transform = 'translateY(0) scale(1) rotateX(0) rotateY(0)';
                el.style.boxShadow = '0 20px 30px -10px rgba(0,0,0,0.2)';
            });
        });

        // --- HIỆU ỨNG XUẤT HIỆN KHI CUỘN & KÍCH HOẠT ĐẾM SỐ ---
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    
                    // Kích hoạt đếm số nếu có data-target
                    const countUpElements = entry.target.querySelectorAll('[data-target]');
                    countUpElements.forEach(el => {
                        if (!el.dataset.animated) {
                            animateCountUp(el);
                            el.dataset.animated = "true";
                        }
                    });
                    
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        document.querySelectorAll('.reveal').forEach(el => {
            observer.observe(el);
        });
    });