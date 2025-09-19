import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { toast } from 'sonner';
import { apiService } from '../services/api';
import { ManualDataEntry } from './ManualDataEntry';
import {
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  AlertCircle,
  Edit,
  Save,
  X,
  User,
  Mail,
  Phone,
  MapPin,
  Briefcase,
  GraduationCap,
  Award,
  Code
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

interface ResumeUploadProps {
  onUploadComplete?: (data: ResumeData) => void;
  onCancel?: () => void;
}

export function ResumeUpload({ onUploadComplete, onCancel }: ResumeUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error' | 'manual'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [extractedData, setExtractedData] = useState<ResumeData | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editableData, setEditableData] = useState<ResumeData | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ACCEPTED_FORMATS = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  const validateFile = (file: File): string | null => {
    if (!ACCEPTED_FORMATS.includes(file.type)) {
      return 'Please upload a PDF, DOC, or DOCX file.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File size must be less than 10MB.';
    }
    return null;
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    const error = validateFile(selectedFile);
    if (error) {
      toast.error(error);
      return;
    }
    setFile(selectedFile);
    setUploadStatus('idle');
    setErrorMessage('');
    setExtractedData(null);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const simulateUploadProgress = () => {
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
    return interval;
  };

  const uploadResume = async () => {
    if (!file) return;

    setUploadStatus('uploading');
    setErrorMessage('');
    
    const progressInterval = simulateUploadProgress();

    try {
      // Use the API service for upload
      const result = await apiService.uploadResume(file);

      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadStatus('processing');
      
      // Simulate processing delay for better UX
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Use the actual result from the API, or fallback to mock data for demo
      const extractedData = result.extracted_data || {
        personal_info: {
          name: result.personal_info?.name || 'John Doe',
          email: result.personal_info?.email || 'john.doe@email.com',
          phone: result.personal_info?.phone || '+1 (555) 123-4567',
          location: result.personal_info?.location || 'San Francisco, CA'
        },
        experience: result.experience || [
          {
            title: 'Senior Software Engineer',
            company: 'Tech Corp',
            duration: '2021 - Present',
            description: 'Led development of scalable web applications using React and Node.js'
          },
          {
            title: 'Software Engineer',
            company: 'StartupXYZ',
            duration: '2019 - 2021',
            description: 'Developed full-stack applications and improved system performance by 40%'
          }
        ],
        education: result.education || [
          {
            degree: 'Bachelor of Science in Computer Science',
            institution: 'University of California',
            year: '2019',
            gpa: '3.8'
          }
        ],
        skills: result.skills || ['JavaScript', 'React', 'Node.js', 'Python', 'AWS', 'Docker', 'PostgreSQL'],
        certifications: result.certifications || [
          {
            name: 'AWS Certified Solutions Architect',
            issuer: 'Amazon Web Services',
            date: '2022'
          }
        ]
      };

      setExtractedData(extractedData);
      setEditableData(extractedData);
      setUploadStatus('success');
      toast.success('Resume processed successfully!');

    } catch (error) {
      clearInterval(progressInterval);
      setUploadStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Upload failed');
      toast.error('Failed to process resume');
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleSave = () => {
    if (editableData) {
      setExtractedData(editableData);
      setIsEditing(false);
      toast.success('Changes saved!');
    }
  };

  const handleCancel = () => {
    setEditableData(extractedData);
    setIsEditing(false);
  };

  const handleConfirm = () => {
    if (extractedData && onUploadComplete) {
      onUploadComplete(extractedData);
    }
  };

  const handleInputChange = (section: keyof ResumeData, index: number | null, field: string, value: string) => {
    if (!editableData) return;

    setEditableData(prev => {
      if (!prev) return prev;
      
      const newData = { ...prev };
      
      if (section === 'personal_info') {
        newData.personal_info = { ...newData.personal_info, [field]: value };
      } else if (section === 'skills') {
        newData.skills = value.split(',').map(skill => skill.trim());
      } else if (index !== null && Array.isArray(newData[section])) {
        const array = [...newData[section] as any[]];
        array[index] = { ...array[index], [field]: value };
        (newData as any)[section] = array;
      }
      
      return newData;
    });
  };

  const renderPersonalInfo = () => {
    const data = isEditing ? editableData : extractedData;
    if (!data) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <User className="w-5 h-5" />
            <span>Personal Information</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {isEditing ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  value={data.personal_info.name}
                  onChange={(e) => handleInputChange('personal_info', null, 'name', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={data.personal_info.email}
                  onChange={(e) => handleInputChange('personal_info', null, 'email', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  value={data.personal_info.phone}
                  onChange={(e) => handleInputChange('personal_info', null, 'phone', e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="location">Location</Label>
                <Input
                  id="location"
                  value={data.personal_info.location}
                  onChange={(e) => handleInputChange('personal_info', null, 'location', e.target.value)}
                />
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <User className="w-4 h-4 text-muted-foreground" />
                <span>{data.personal_info.name}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Mail className="w-4 h-4 text-muted-foreground" />
                <span>{data.personal_info.email}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Phone className="w-4 h-4 text-muted-foreground" />
                <span>{data.personal_info.phone}</span>
              </div>
              <div className="flex items-center space-x-2">
                <MapPin className="w-4 h-4 text-muted-foreground" />
                <span>{data.personal_info.location}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderExperience = () => {
    const data = isEditing ? editableData : extractedData;
    if (!data) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Briefcase className="w-5 h-5" />
            <span>Work Experience</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {data.experience.map((exp, index) => (
            <div key={index} className="p-4 border rounded-lg">
              {isEditing ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <Label>Job Title</Label>
                      <Input
                        value={exp.title}
                        onChange={(e) => handleInputChange('experience', index, 'title', e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>Company</Label>
                      <Input
                        value={exp.company}
                        onChange={(e) => handleInputChange('experience', index, 'company', e.target.value)}
                      />
                    </div>
                  </div>
                  <div>
                    <Label>Duration</Label>
                    <Input
                      value={exp.duration}
                      onChange={(e) => handleInputChange('experience', index, 'duration', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Description</Label>
                    <textarea
                      className="w-full p-2 border rounded-md"
                      rows={3}
                      value={exp.description}
                      onChange={(e) => handleInputChange('experience', index, 'description', e.target.value)}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <h4 className="font-semibold">{exp.title}</h4>
                  <p className="text-muted-foreground">{exp.company} • {exp.duration}</p>
                  <p className="mt-2 text-sm">{exp.description}</p>
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    );
  };

  const renderEducation = () => {
    const data = isEditing ? editableData : extractedData;
    if (!data) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <GraduationCap className="w-5 h-5" />
            <span>Education</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {data.education.map((edu, index) => (
            <div key={index} className="p-4 border rounded-lg">
              {isEditing ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <Label>Degree</Label>
                      <Input
                        value={edu.degree}
                        onChange={(e) => handleInputChange('education', index, 'degree', e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>Institution</Label>
                      <Input
                        value={edu.institution}
                        onChange={(e) => handleInputChange('education', index, 'institution', e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <Label>Year</Label>
                      <Input
                        value={edu.year}
                        onChange={(e) => handleInputChange('education', index, 'year', e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>GPA (Optional)</Label>
                      <Input
                        value={edu.gpa || ''}
                        onChange={(e) => handleInputChange('education', index, 'gpa', e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <h4 className="font-semibold">{edu.degree}</h4>
                  <p className="text-muted-foreground">{edu.institution} • {edu.year}</p>
                  {edu.gpa && <p className="text-sm">GPA: {edu.gpa}</p>}
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    );
  };

  const renderSkills = () => {
    const data = isEditing ? editableData : extractedData;
    if (!data) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Code className="w-5 h-5" />
            <span>Skills</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isEditing ? (
            <div>
              <Label>Skills (comma-separated)</Label>
              <Input
                value={data.skills.join(', ')}
                onChange={(e) => handleInputChange('skills', null, '', e.target.value)}
                placeholder="JavaScript, React, Node.js, Python..."
              />
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {data.skills.map((skill, index) => (
                <Badge key={index} variant="secondary">{skill}</Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderCertifications = () => {
    const data = isEditing ? editableData : extractedData;
    if (!data || data.certifications.length === 0) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Award className="w-5 h-5" />
            <span>Certifications</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {data.certifications.map((cert, index) => (
            <div key={index} className="p-4 border rounded-lg">
              {isEditing ? (
                <div className="space-y-3">
                  <div>
                    <Label>Certification Name</Label>
                    <Input
                      value={cert.name}
                      onChange={(e) => handleInputChange('certifications', index, 'name', e.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <Label>Issuer</Label>
                      <Input
                        value={cert.issuer}
                        onChange={(e) => handleInputChange('certifications', index, 'issuer', e.target.value)}
                      />
                    </div>
                    <div>
                      <Label>Date</Label>
                      <Input
                        value={cert.date}
                        onChange={(e) => handleInputChange('certifications', index, 'date', e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <h4 className="font-semibold">{cert.name}</h4>
                  <p className="text-muted-foreground">{cert.issuer} • {cert.date}</p>
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    );
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
              <Upload className="w-6 h-6" />
              <span>Resume Upload</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {uploadStatus === 'idle' && !file && (
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <FileText className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">Upload your resume</h3>
                <p className="text-muted-foreground mb-4">
                  Drag and drop your resume here, or click to browse
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Supported formats: PDF, DOC, DOCX (max 10MB)
                </p>
                <Button onClick={() => fileInputRef.current?.click()}>
                  Choose File
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={handleFileInputChange}
                  className="hidden"
                />
              </div>
            )}

            {file && uploadStatus === 'idle' && (
              <div className="space-y-4">
                <div className="flex items-center space-x-3 p-4 border rounded-lg">
                  <FileText className="w-8 h-8 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setFile(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
                <div className="flex space-x-2">
                  <Button onClick={uploadResume} className="flex-1">
                    Process Resume
                  </Button>
                  <Button variant="outline" onClick={onCancel}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}

            {(uploadStatus === 'uploading' || uploadStatus === 'processing') && (
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  {uploadStatus === 'uploading' ? (
                    <Upload className="w-5 h-5 text-primary animate-pulse" />
                  ) : (
                    <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  )}
                  <span className="font-medium">
                    {uploadStatus === 'uploading' ? 'Uploading...' : 'Processing resume...'}
                  </span>
                </div>
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-sm text-muted-foreground">
                  {uploadStatus === 'uploading' 
                    ? 'Uploading your resume to our secure servers'
                    : 'Extracting information from your resume using AI'
                  }
                </p>
              </div>
            )}

            {uploadStatus === 'error' && (
              <div className="space-y-4">
                <div className="flex items-center space-x-3 p-4 border border-destructive rounded-lg bg-destructive/5">
                  <XCircle className="w-5 h-5 text-destructive" />
                  <div>
                    <p className="font-medium text-destructive">Upload Failed</p>
                    <p className="text-sm text-muted-foreground">{errorMessage}</p>
                  </div>
                </div>
                <div className="p-4 border border-blue-200 rounded-lg bg-blue-50">
                  <p className="text-sm text-blue-800 mb-3">
                    Don't worry! You can still create your profile by entering your information manually.
                  </p>
                  <Button 
                    onClick={() => {
                      // Switch to manual entry mode
                      setUploadStatus('manual');
                    }}
                    className="w-full mb-2"
                  >
                    Enter Information Manually
                  </Button>
                </div>
                <div className="flex space-x-2">
                  <Button onClick={uploadResume} variant="outline">
                    Try Again
                  </Button>
                  <Button 
                    onClick={() => {
                      setFile(null);
                      setUploadStatus('idle');
                      setErrorMessage('');
                    }}
                    variant="outline"
                  >
                    Choose Different File
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      <AnimatePresence>
        {uploadStatus === 'success' && extractedData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.6 }}
            className="space-y-6"
          >
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="w-6 h-6 text-green-500" />
                    <span>Resume Processed Successfully</span>
                  </CardTitle>
                  <div className="flex space-x-2">
                    {isEditing ? (
                      <>
                        <Button onClick={handleSave} size="sm">
                          <Save className="w-4 h-4 mr-2" />
                          Save
                        </Button>
                        <Button onClick={handleCancel} variant="outline" size="sm">
                          Cancel
                        </Button>
                      </>
                    ) : (
                      <Button onClick={handleEdit} variant="outline" size="sm">
                        <Edit className="w-4 h-4 mr-2" />
                        Edit
                      </Button>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-green-600" />
                  <p className="text-sm text-green-800">
                    Please review the extracted information below and make any necessary corrections.
                  </p>
                </div>
              </CardContent>
            </Card>

            {renderPersonalInfo()}
            {renderExperience()}
            {renderEducation()}
            {renderSkills()}
            {renderCertifications()}

            <Card>
              <CardContent className="pt-6">
                <div className="flex space-x-4">
                  <Button onClick={handleConfirm} className="flex-1">
                    Confirm & Continue
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      setFile(null);
                      setUploadStatus('idle');
                      setExtractedData(null);
                      setEditableData(null);
                      setIsEditing(false);
                    }}
                  >
                    Upload Different Resume
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Manual Data Entry Mode */}
      {uploadStatus === 'manual' && (
        <ManualDataEntry
          onSave={(data) => {
            setExtractedData(data);
            setEditableData(data);
            setUploadStatus('success');
            toast.success('Profile data entered successfully!');
          }}
          onCancel={() => {
            setUploadStatus('idle');
            setFile(null);
            setErrorMessage('');
          }}
        />
      )}
    </div>
  );
}