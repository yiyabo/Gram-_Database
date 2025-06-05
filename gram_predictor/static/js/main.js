document.addEventListener('DOMContentLoaded', () => {
    if (typeof tsParticles !== 'undefined') {
        tsParticles.load("tsparticles-background", {
            fpsLimit: 60,
            background: {
                color: "transparent" // CSS already sets body background
            },
            particles: {
                number: {
                    value: 100, // Adjust for density
                    density: {
                        enable: true,
                        area: 800
                    }
                },
                color: {
                    value: ["#333333", "#666666", "#999999", "#00bcd4", "#ffffff"] // Greys, cyan, white
                },
                shape: {
                    type: "square", // Small squares
                },
                opacity: {
                    value: { min: 0.1, max: 0.6 }, // Random opacity
                    animation: {
                        enable: true,
                        speed: 0.5,
                        minimumValue: 0.1,
                        sync: false
                    }
                },
                size: {
                    value: { min: 1, max: 4 }, // Random size
                    animation: {
                        enable: true,
                        speed: 2,
                        minimumValue: 0.5,
                        sync: false
                    }
                },
                links: {
                    enable: true,
                    distance: 120, // Max distance for linking
                    color: "#444444", // Link color
                    opacity: 0.3,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 0.8, // Slower movement
                    direction: "none", // Particles move in random directions
                    random: true,
                    straight: false,
                    outModes: {
                        default: "out" // Particles go out of canvas
                    },
                    attract: { // Subtle attraction to create clusters or central pull
                        enable: false, // Keep false for now, can enable for different effect
                        rotateX: 600,
                        rotateY: 1200
                    },
                    // To achieve a more "orbiting" or "galaxy" feel:
                    // We can try to make particles orbit around the center.
                    // This is a simplified approach. For true 3D, more complex pathing or orbit config is needed.
                    // A more direct orbit config might be:
                    /*
                    orbit: {
                        enable: true,
                        opacity: 0.1,
                        rotation: {
                            random: {
                                enable: true
                            },
                            value: 45 // degrees
                        },
                        width: 0.5,
                        radius: undefined // particles will have random radius
                    }
                    */
                   // For a start, let's use a general outward/random movement.
                   // True 3D rotation of the entire canvas/particle system is harder with 2D canvas.
                   // We can simulate depth with size/speed variations.
                }
            },
            interactivity: {
                detectsOn: "canvas",
                events: {
                    onHover: {
                        enable: true,
                        mode: "repulse" // Particles move away from cursor
                    },
                    onClick: {
                        enable: true,
                        mode: "push" // Push new particles on click
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
