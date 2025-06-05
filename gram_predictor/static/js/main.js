document.addEventListener('DOMContentLoaded', () => {
    if (typeof tsParticles !== 'undefined') {
        tsParticles.load("tsparticles-background", {
            fpsLimit: 60,
            background: {
                color: "transparent" // Body background is set in CSS
            },
            particles: {
                number: {
                    value: 100, // Restored value
                    density: {
                        enable: true,
                        area: 800 // Restored value
                    }
                },
                color: {
                    value: ["#333333", "#666666", "#999999", "#00E5FF", "#FFFFFF"] // Restored: Greys, accent-cyan, white
                },
                shape: {
                    type: "square", // Restored shape
                },
                opacity: {
                    value: { min: 0.1, max: 0.6 }, // Random opacity
                    animation: {
                        enable: true,
                        speed: 0.5, // Restored speed
                        minimumValue: 0.1, // Restored min value
                        sync: false
                    }
                },
                size: {
                    value: { min: 1, max: 4 }, // Restored size
                    animation: {
                        enable: true,
                        speed: 2, // Restored speed
                        minimumValue: 0.5, // Restored min value
                        sync: false
                    }
                },
                links: {
                    enable: true,
                    distance: 120, // Restored distance
                    color: "#444444", // Restored link color (can be var(--glass-border-color) later)
                    opacity: 0.3, // Restored opacity
                    width: 1 // Restored width
                },
                move: {
                    enable: true,
                    speed: 0.8, // Restored speed
                    direction: "none",
                    random: true,
                    straight: false,
                    outModes: {
                        default: "out"
                    },
                    attract: {
                        enable: false // Kept false as per original restored config
                    }
                    // Removed wobble and noise from Nebula Haze
                }
            },
            interactivity: {
                detectsOn: "canvas",
                events: {
                    onHover: {
                        enable: true,
                        mode: "repulse" // Restored mode
                    },
                    onClick: {
                        enable: true, // Restored enable state
                        mode: "push" // Restored mode
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 150,
                        links: {
                            opacity: 0.5
                        }
                    },
                    bubble: {
                        distance: 200,
                        size: 10,
                        duration: 2,
                        opacity: 0.8
                    },
                    repulse: {
                        distance: 80, // Repulse distance
                        duration: 0.4
                    },
                    push: {
                        quantity: 3 // Number of particles to push on click
                    },
                    remove: {
                        quantity: 2
                    }
                }
            },
            detectRetina: true // For high-density displays
        }).then(container => {
            console.log("tsParticles loaded successfully");
            // You can further interact with the container here if needed
            // For example, to simulate rotation, you might try to slowly update
            // a global angle and re-calculate particle positions if using custom paths,
            // or adjust some global movement parameters if the library allows.
            // tsParticles itself doesn't have a simple "rotate entire scene" for 2D canvas.
            // The "orbit" feature is per-particle.
        }).catch(error => {
            console.error("Error loading tsParticles:", error);
        });
    } else {
        console.error("tsParticles library not found.");
    }
});
