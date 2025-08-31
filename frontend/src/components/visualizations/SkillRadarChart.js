'use client';

import React, { useRef, useEffect, useState } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip
} from 'recharts';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

const SkillRadarChart = ({ 
  skills = [], 
  title = "Skill Profile",
  showComparison = false,
  comparisonData = null,
  useD3 = false,
  className = ""
}) => {
  const d3Ref = useRef();
  const [animationComplete, setAnimationComplete] = useState(false);

  // Transform skills data for radar chart
  const chartData = skills.map(skill => ({
    skill: skill.name,
    current: Math.round(skill.level * 100),
    required: skill.required ? Math.round(skill.required * 100) : null,
    fullMark: 100
  }));

  // D3.js Radar Chart Implementation
  useEffect(() => {
    if (!useD3 || !d3Ref.current || skills.length === 0) return;

    const svg = d3.select(d3Ref.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 400;
    const margin = 60;
    const radius = Math.min(width, height) / 2 - margin;

    const g = svg
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2}, ${height / 2})`);

    // Create scales
    const angleScale = d3.scaleLinear()
      .domain([0, skills.length])
      .range([0, 2 * Math.PI]);

    const radiusScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, radius]);

    // Draw grid circles
    const gridLevels = 5;
    for (let i = 1; i <= gridLevels; i++) {
      g.append("circle")
        .attr("r", (radius / gridLevels) * i)
        .attr("fill", "none")
        .attr("stroke", "#e5e7eb")
        .attr("stroke-width", 1)
        .attr("opacity", 0.5);
    }

    // Draw grid lines
    skills.forEach((_, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", radius * Math.cos(angle))
        .attr("y2", radius * Math.sin(angle))
        .attr("stroke", "#e5e7eb")
        .attr("stroke-width", 1)
        .attr("opacity", 0.5);
    });

    // Create path generator
    const line = d3.line()
      .x((d, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        return radiusScale(d.current) * Math.cos(angle);
      })
      .y((d, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        return radiusScale(d.current) * Math.sin(angle);
      })
      .curve(d3.curveLinearClosed);

    // Draw skill area
    const path = g.append("path")
      .datum(chartData)
      .attr("d", line)
      .attr("fill", "#3b82f6")
      .attr("fill-opacity", 0.3)
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2);

    // Animate path
    const totalLength = path.node().getTotalLength();
    path
      .attr("stroke-dasharray", totalLength + " " + totalLength)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(1500)
      .ease(d3.easeLinear)
      .attr("stroke-dashoffset", 0)
      .on("end", () => setAnimationComplete(true));

    // Add skill points
    chartData.forEach((d, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const x = radiusScale(d.current) * Math.cos(angle);
      const y = radiusScale(d.current) * Math.sin(angle);

      g.append("circle")
        .attr("cx", x)
        .attr("cy", y)
        .attr("r", 0)
        .attr("fill", "#3b82f6")
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 2)
        .transition()
        .delay(1500 + i * 100)
        .duration(300)
        .attr("r", 4);
    });

    // Add skill labels
    skills.forEach((skill, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const labelRadius = radius + 20;
      const x = labelRadius * Math.cos(angle);
      const y = labelRadius * Math.sin(angle);

      g.append("text")
        .attr("x", x)
        .attr("y", y)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "12px")
        .attr("fill", "#6b7280")
        .attr("opacity", 0)
        .text(skill.name)
        .transition()
        .delay(2000 + i * 100)
        .duration(300)
        .attr("opacity", 1);
    });

  }, [useD3, skills, chartData]);

  useEffect(() => {
    if (!useD3) {
      const timer = setTimeout(() => setAnimationComplete(true), 1000);
      return () => clearTimeout(timer);
    }
  }, [useD3]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="font-semibold text-gray-900 dark:text-white">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey === 'current' ? 'Current Level' : 'Required Level'}: {entry.value}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 ${className}`}
    >
      <motion.h3
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-xl font-bold text-gray-900 dark:text-white mb-6 text-center"
      >
        {title}
      </motion.h3>
      
      <div className="h-96 flex justify-center">
        {useD3 ? (
          <svg ref={d3Ref} className="max-w-full max-h-full" />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <PolarGrid 
                stroke="#e5e7eb" 
                className="dark:stroke-gray-600"
              />
              <PolarAngleAxis 
                dataKey="skill" 
                tick={{ fontSize: 12, fill: '#6b7280' }}
                className="dark:fill-gray-300"
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]} 
                tick={{ fontSize: 10, fill: '#9ca3af' }}
                className="dark:fill-gray-400"
              />
              
              <Radar
                name="Current Level"
                dataKey="current"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
              
              {showComparison && (
                <Radar
                  name="Required Level"
                  dataKey="required"
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.1}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ fill: '#ef4444', strokeWidth: 2, r: 3 }}
                />
              )}
              
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                iconType="line"
              />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </div>
      
      {/* Skill level indicators */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-4 flex flex-wrap gap-2 justify-center"
      >
        {skills.slice(0, 5).map((skill, index) => (
          <motion.div
            key={skill.name}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 + index * 0.1 }}
            className="flex items-center gap-2 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded-full"
          >
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: skill.level > 0.7 ? '#10b981' : skill.level > 0.4 ? '#f59e0b' : '#ef4444' }}
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {skill.name}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {Math.round(skill.level * 100)}%
            </span>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
};

export default SkillRadarChart;