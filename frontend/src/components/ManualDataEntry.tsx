import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { toast } from 'sonner';
import {
  User,
  Mail,
  Phone,
  MapPin,
  Briefcase,
  GraduationCap,
  Award,
  Code,
  Plus,
  Trash2,
  Save
} from 'lucide-react';

interface ResumeData {
  personal_info: {
    name: string;
    email: string;
    phone: string;
    location: string;
  };
  experience: Array<{
    title: string;
    company: string;
    duration: string;
    description: string;
  }>;
  education: Array<{
    degree: string;
    institution: string;
    year: string;
    gpa?: string;
  }>;
  skills: string[];
  certifications: Array<{
    name: string;
    issuer: string;
    date: string;
  }>;
}

interface ManualDataEntryProps {
  initialData?: Partial<ResumeData>;
  onSave: (data: ResumeData) => void;
  onCancel: () => void;
}

export function ManualDataEntry({ initialData, onSave, onCancel }: ManualDataEntryProps) {
  const [formData, setFormData] = useState<ResumeData>({
    personal_info: {
      name: initialData?.personal_info?.name || '',
      email: initialData?.personal_info?.email || '',
      phone: initialData?.personal_info?.phone || '',
      location: initialData?.personal_info?.location || '',
    },
    experience: initialData?.experience || [
      {
        title: '',
        company: '',
        duration: '',
        description: '',
      }
    ],
    education: initialData?.education || [
      {
        degree: '',
        institution: '',
        year: '',
        gpa: '',
      }
    ],
    skills: initialData?.skills || [],
    certifications: initialData?.certifications || [],
  });

  const [skillsInput, setSkillsInput] = useState(
    initialData?.skills?.join(', ') || ''
  );

  const handlePersonalInfoChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      personal_info: {
        ...prev.personal_info,
        [field]: value,
      }
    }));
  };

  const handleExperienceChange = (index: number, field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      experience: prev.experience.map((exp, i) => 
        i === index ? { ...exp, [field]: value } : exp
      )
    }));
  };

  const addExperience = () => {
    setFormData(prev => ({
      ...prev,
      experience: [
        ...prev.experience,
        {
          title: '',
          company: '',
          duration: '',
          description: '',
        }
      ]
    }));
  };

  const removeExperience = (index: number) => {
    if (formData.experience.length > 1) {
      setFormData(prev => ({
        ...prev,
        experience: prev.experience.filter((_, i) => i !== index)
      }));
    }
  };

  const handleEducationChange = (index: number, field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      education: prev.education.map((edu, i) => 
        i === index ? { ...edu, [field]: value } : edu
      )
    }));
  };

  const addEducation = () => {
    setFormData(prev => ({
      ...prev,
      education: [
        ...prev.education,
        {
          degree: '',
          institution: '',
          year: '',
          gpa: '',
        }
      ]
    }));
  };

  const removeEducation = (index: number) => {
    if (formData.education.length > 1) {
      setFormData(prev => ({
        ...prev,
        education: prev.education.filter((_, i) => i !== index)
      }));
    }
  };

  const handleSkillsChange = (value: string) => {
    setSkillsInput(value);
    const skills = value.split(',').map(skill => skill.trim()).filter(skill => skill.length > 0);
    setFormData(prev => ({
      ...prev,
      skills
    }));
  };

  const handleCertificationChange = (index: number, field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      certifications: prev.certifications.map((cert, i) => 
        i === index ? { ...cert, [field]: value } : cert
      )
    }));
  };

  const addCertification = () => {
    setFormData(prev => ({
      ...prev,
      certifications: [
        ...prev.certifications,
        {
          name: '',
          issuer: '',
          date: '',
        }
      ]
    }));
  };

  const removeCertification = (index: number) => {
    setFormData(prev => ({
      ...prev,
      certifications: prev.certifications.filter((_, i) => i !== index)
    }));
  };

  const validateForm = (): boolean => {
    if (!formData.personal_info.name.trim()) {
      toast.error('Please enter your name');
      return false;
    }
    if (!formData.personal_info.email.trim()) {
      toast.error('Please enter your email');
      return false;
    }
    if (formData.experience.some(exp => !exp.title.trim() || !exp.company.trim())) {
      toast.error('Please fill in all experience fields');
      return false;
    }
    if (formData.education.some(edu => !edu.degree.trim() || !edu.institution.trim())) {
      toast.error('Please fill in all education fields');
      return false;
    }
    return true;
  };

  const handleSave = () => {
    if (validateForm()) {
      onSave(formData);
      toast.success('Profile data saved successfully!');
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="w-6 h-6" />
              <span>Manual Profile Entry</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Please fill in your profile information manually. All fields marked with * are required.
            </p>
          </CardContent>
        </Card>
      </motion.div>

      {/* Personal Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="w-5 h-5" />
              <span>Personal Information</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name">Full Name *</Label>
                <Input
                  id="name"
                  value={formData.personal_info.name}
                  onChange={(e) => handlePersonalInfoChange('name', e.target.value)}
                  placeholder="Enter your full name"
                />
              </div>
              <div>
                <Label htmlFor="email">Email *</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.personal_info.email}
                  onChange={(e) => handlePersonalInfoChange('email', e.target.value)}
                  placeholder="Enter your email address"
                />
              </div>
              <div>
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  value={formData.personal_info.phone}
                  onChange={(e) => handlePersonalInfoChange('phone', e.target.value)}
                  placeholder="Enter your phone number"
                />
              </div>
              <div>
                <Label htmlFor="location">Location</Label>
                <Input
                  id="location"
                  value={formData.personal_info.location}
                  onChange={(e) => handlePersonalInfoChange('location', e.target.value)}
                  placeholder="Enter your location"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Work Experience */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2">
                <Briefcase className="w-5 h-5" />
                <span>Work Experience</span>
              </CardTitle>
              <Button onClick={addExperience} size="sm" variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Add Experience
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {formData.experience.map((exp, index) => (
              <div key={index} className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Experience {index + 1}</h4>
                  {formData.experience.length > 1 && (
                    <Button
                      onClick={() => removeExperience(index)}
                      size="sm"
                      variant="ghost"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Job Title *</Label>
                    <Input
                      value={exp.title}
                      onChange={(e) => handleExperienceChange(index, 'title', e.target.value)}
                      placeholder="e.g., Software Engineer"
                    />
                  </div>
                  <div>
                    <Label>Company *</Label>
                    <Input
                      value={exp.company}
                      onChange={(e) => handleExperienceChange(index, 'company', e.target.value)}
                      placeholder="e.g., Tech Corp"
                    />
                  </div>
                </div>
                <div>
                  <Label>Duration</Label>
                  <Input
                    value={exp.duration}
                    onChange={(e) => handleExperienceChange(index, 'duration', e.target.value)}
                    placeholder="e.g., Jan 2020 - Present"
                  />
                </div>
                <div>
                  <Label>Description</Label>
                  <textarea
                    className="w-full p-3 border border-input rounded-md resize-none"
                    rows={3}
                    value={exp.description}
                    onChange={(e) => handleExperienceChange(index, 'description', e.target.value)}
                    placeholder="Describe your responsibilities and achievements..."
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </motion.div>

      {/* Education */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2">
                <GraduationCap className="w-5 h-5" />
                <span>Education</span>
              </CardTitle>
              <Button onClick={addEducation} size="sm" variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Add Education
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {formData.education.map((edu, index) => (
              <div key={index} className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Education {index + 1}</h4>
                  {formData.education.length > 1 && (
                    <Button
                      onClick={() => removeEducation(index)}
                      size="sm"
                      variant="ghost"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Degree *</Label>
                    <Input
                      value={edu.degree}
                      onChange={(e) => handleEducationChange(index, 'degree', e.target.value)}
                      placeholder="e.g., Bachelor of Science in Computer Science"
                    />
                  </div>
                  <div>
                    <Label>Institution *</Label>
                    <Input
                      value={edu.institution}
                      onChange={(e) => handleEducationChange(index, 'institution', e.target.value)}
                      placeholder="e.g., University of California"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Year</Label>
                    <Input
                      value={edu.year}
                      onChange={(e) => handleEducationChange(index, 'year', e.target.value)}
                      placeholder="e.g., 2019"
                    />
                  </div>
                  <div>
                    <Label>GPA (Optional)</Label>
                    <Input
                      value={edu.gpa || ''}
                      onChange={(e) => handleEducationChange(index, 'gpa', e.target.value)}
                      placeholder="e.g., 3.8"
                    />
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </motion.div>

      {/* Skills */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Code className="w-5 h-5" />
              <span>Skills</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="skills">Skills (comma-separated)</Label>
              <Input
                id="skills"
                value={skillsInput}
                onChange={(e) => handleSkillsChange(e.target.value)}
                placeholder="e.g., JavaScript, React, Node.js, Python, AWS"
              />
            </div>
            {formData.skills.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {formData.skills.map((skill, index) => (
                  <Badge key={index} variant="secondary">{skill}</Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Certifications */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2">
                <Award className="w-5 h-5" />
                <span>Certifications (Optional)</span>
              </CardTitle>
              <Button onClick={addCertification} size="sm" variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Add Certification
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {formData.certifications.length === 0 ? (
              <p className="text-muted-foreground text-center py-4">
                No certifications added yet. Click "Add Certification" to get started.
              </p>
            ) : (
              formData.certifications.map((cert, index) => (
                <div key={index} className="p-4 border rounded-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">Certification {index + 1}</h4>
                    <Button
                      onClick={() => removeCertification(index)}
                      size="sm"
                      variant="ghost"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  <div>
                    <Label>Certification Name</Label>
                    <Input
                      value={cert.name}
                      onChange={(e) => handleCertificationChange(index, 'name', e.target.value)}
                      placeholder="e.g., AWS Certified Solutions Architect"
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>Issuer</Label>
                      <Input
                        value={cert.issuer}
                        onChange={(e) => handleCertificationChange(index, 'issuer', e.target.value)}
                        placeholder="e.g., Amazon Web Services"
                      />
                    </div>
                    <div>
                      <Label>Date</Label>
                      <Input
                        value={cert.date}
                        onChange={(e) => handleCertificationChange(index, 'date', e.target.value)}
                        placeholder="e.g., 2022"
                      />
                    </div>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <Card>
          <CardContent className="pt-6">
            <div className="flex space-x-4">
              <Button onClick={handleSave} className="flex-1">
                <Save className="w-4 h-4 mr-2" />
                Save Profile
              </Button>
              <Button variant="outline" onClick={onCancel}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}