document.addEventListener("DOMContentLoaded", function() {

    // === FIX RELOAD JUMP & IMPLEMENT SMOOTH SCROLL ===

    // 1. Ngăn trình duyệt tự nhảy đến anchor khi reload
    if (history.scrollRestoration) {
        history.scrollRestoration = 'manual';
    }

    // 2. Xử lý click cho các link anchor để cuộn mượt và làm sạch URL
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault(); // Ngăn hành vi nhảy trang mặc định

            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                // Thực hiện cuộn mượt
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });

                // Sau khi cuộn, làm sạch URL để tránh lỗi reload
                // Dùng setTimeout để đảm bảo việc thay đổi URL xảy ra sau khi cuộn đã bắt đầu
                setTimeout(() => {
                    history.replaceState(null, null, ' ');
                }, 800); // 800ms là đủ thời gian cho cuộn
            }
        });
    });


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

    // --- HIỆU ỨNG XUẤT HIỆN KHI CUỘN & KÍCH HOẠT ĐẾM SỐ ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                
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