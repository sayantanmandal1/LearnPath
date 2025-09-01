'use client';

import React, { useEffect, useRef } from 'react';
import anime from 'animejs';

const AnimatedBackground = ({ variant = 'particles', className = '' }) => {
  const containerRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    
    // Clear any existing content
    container.innerHTML = '';

    if (variant === 'particles') {
      createParticleAnimation(container);
    } else if (variant === 'waves') {
      createWaveAnimation(container);
    } else if (variant === 'geometric') {
      createGeometricAnimation(container);
    } else if (variant === 'neural') {
      createNeuralNetworkAnimation(container);
    }

    return () => {
      if (animationRef.current) {
        animationRef.current.pause();
      }
    };
  }, [variant]);

  const createParticleAnimation = (container) => {
    const particleCount = 50;
    const particles = [];

    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.className = 'absolute rounded-full bg-gradient-to-r from-primary-400 to-secondary-400 opacity-20';
      
      const size = Math.random() * 4 + 2;
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.top = `${Math.random() * 100}%`;
      
      container.appendChild(particle);
      particles.push(particle);
    }

    animationRef.current = anime({
      targets: particles,
      translateX: () => anime.random(-100, 100),
      translateY: () => anime.random(-100, 100),
      scale: () => anime.random(0.5, 2),
      opacity: () => anime.random(0.1, 0.8),
      duration: () => anime.random(3000, 8000),
      easing: 'easeInOutSine',
      loop: true,
      direction: 'alternate',
      delay: anime.stagger(100)
    });
  };

  const createWaveAnimation = (container) => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 1200 320');
    svg.setAttribute('class', 'absolute inset-0 w-full h-full');
    
    const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path1.setAttribute('fill', 'url(#gradient1)');
    path1.setAttribute('fill-opacity', '0.1');
    
    const path2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path2.setAttribute('fill', 'url(#gradient2)');
    path2.setAttribute('fill-opacity', '0.1');
    
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient1 = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient1.setAttribute('id', 'gradient1');
    gradient1.innerHTML = `
      <stop offset="0%" style="stop-color:#0ea5e9;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#d946ef;stop-opacity:1" />
    `;
    
    const gradient2 = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient2.setAttribute('id', 'gradient2');
    gradient2.innerHTML = `
      <stop offset="0%" style="stop-color:#f97316;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0ea5e9;stop-opacity:1" />
    `;
    
    defs.appendChild(gradient1);
    defs.appendChild(gradient2);
    svg.appendChild(defs);
    svg.appendChild(path1);
    svg.appendChild(path2);
    container.appendChild(svg);

    const animateWaves = () => {
      const time = Date.now() * 0.001;
      
      const wave1 = `M0,160L48,144C96,128,192,96,288,106.7C384,117,480,171,576,186.7C672,203,768,181,864,154.7C960,128,1056,96,1152,90.7C1248,85,1344,107,1392,117.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z`;
      
      const wave2 = `M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,208C672,213,768,203,864,181.3C960,160,1056,128,1152,128C1248,128,1344,160,1392,176L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z`;
      
      path1.setAttribute('d', wave1);
      path2.setAttribute('d', wave2);
      
      requestAnimationFrame(animateWaves);
    };
    
    animateWaves();
  };

  const createGeometricAnimation = (container) => {
    const shapes = ['circle', 'triangle', 'square', 'hexagon'];
    const shapeElements = [];

    for (let i = 0; i < 20; i++) {
      const shape = document.createElement('div');
      const shapeType = shapes[Math.floor(Math.random() * shapes.length)];
      
      shape.className = `absolute opacity-10 border-2 border-primary-400`;
      
      const size = Math.random() * 60 + 20;
      shape.style.width = `${size}px`;
      shape.style.height = `${size}px`;
      shape.style.left = `${Math.random() * 100}%`;
      shape.style.top = `${Math.random() * 100}%`;
      
      if (shapeType === 'circle') {
        shape.style.borderRadius = '50%';
      } else if (shapeType === 'triangle') {
        shape.style.clipPath = 'polygon(50% 0%, 0% 100%, 100% 100%)';
      } else if (shapeType === 'hexagon') {
        shape.style.clipPath = 'polygon(30% 0%, 70% 0%, 100% 50%, 70% 100%, 30% 100%, 0% 50%)';
      }
      
      container.appendChild(shape);
      shapeElements.push(shape);
    }

    animationRef.current = anime({
      targets: shapeElements,
      rotate: '1turn',
      translateX: () => anime.random(-200, 200),
      translateY: () => anime.random(-200, 200),
      scale: () => anime.random(0.5, 1.5),
      duration: () => anime.random(8000, 15000),
      easing: 'easeInOutQuad',
      loop: true,
      direction: 'alternate',
      delay: anime.stagger(200)
    });
  };

  const createNeuralNetworkAnimation = (container) => {
    const nodeCount = 30;
    const nodes = [];
    const connections = [];

    // Create nodes
    for (let i = 0; i < nodeCount; i++) {
      const node = document.createElement('div');
      node.className = 'absolute w-2 h-2 bg-primary-400 rounded-full opacity-60';
      node.style.left = `${Math.random() * 100}%`;
      node.style.top = `${Math.random() * 100}%`;
      
      container.appendChild(node);
      nodes.push({
        element: node,
        x: parseFloat(node.style.left),
        y: parseFloat(node.style.top)
      });
    }

    // Create SVG for connections
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'absolute inset-0 w-full h-full pointer-events-none');
    container.appendChild(svg);

    // Create connections between nearby nodes
    nodes.forEach((node, i) => {
      nodes.forEach((otherNode, j) => {
        if (i !== j) {
          const distance = Math.sqrt(
            Math.pow(node.x - otherNode.x, 2) + Math.pow(node.y - otherNode.y, 2)
          );
          
          if (distance < 20) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('stroke', '#0ea5e9');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('opacity', '0.2');
            line.setAttribute('x1', `${node.x}%`);
            line.setAttribute('y1', `${node.y}%`);
            line.setAttribute('x2', `${otherNode.x}%`);
            line.setAttribute('y2', `${otherNode.y}%`);
            
            svg.appendChild(line);
            connections.push(line);
          }
        }
      });
    });

    // Animate nodes
    animationRef.current = anime({
      targets: nodes.map(n => n.element),
      scale: [1, 1.5, 1],
      opacity: [0.6, 1, 0.6],
      duration: 3000,
      easing: 'easeInOutSine',
      loop: true,
      delay: anime.stagger(100)
    });

    // Animate connections
    anime({
      targets: connections,
      opacity: [0.2, 0.6, 0.2],
      duration: 4000,
      easing: 'easeInOutSine',
      loop: true,
      delay: anime.stagger(50)
    });
  };

  return (
    <div 
      ref={containerRef}
      className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}
    />
  );
};

export default AnimatedBackground;