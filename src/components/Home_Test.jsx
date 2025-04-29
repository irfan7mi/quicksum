import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Home.css';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import FilePresentIcon from '@mui/icons-material/FilePresent';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import HistoryIcon from '@mui/icons-material/History';
import { toast } from 'react-toastify';
import { useStore } from '../StoreContext';
import SpatialTrackingIcon from '@mui/icons-material/SpatialTracking';
import LogoutIcon from '@mui/icons-material/Logout';

function Home({setIsAuthenticated}) {
  const [inputMethod, setInputMethod] = useState("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState("");
  const [summary, setSummary] = useState("");
  const [summaryLevel, setSummaryLevel] = useState(50);
  const [mode, setMode] = useState("text");
  const [loading, setLoading] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [outputFormat, setOutputFormat] = useState("paragraph");
  const [summaryType, setSummaryType] = useState("");
  const { email} = useStore();
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [isVisible, setIsVisible] = useState(false);

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    
    const fileType = uploadedFile.name.split('.').pop().toLowerCase();

    if (fileType === "txt" || fileType === "pdf") {
        setMode("file"); 
        setInputMethod(fileType === "txt" ? "file" : "pdf");  
    } else if (fileType === "mp3") {
        setMode("audio"); 
    }

    if (fileType === "txt") {
        const reader = new FileReader();
        reader.onload = (e) => {
            setText(e.target.result); 
        };
        reader.readAsText(uploadedFile);
    }
  };
  useEffect(() => {
    if (mode === "file" || mode === "pdf") {
        setInputMethod("file");  
    } else if (mode === "audio") {
        setInputMethod("audio");
    }
  }, [mode]);

  const handleProcessTextSumm = async () => {
    if (!text.trim()) {
        alert("Please enter some text to summarize.");
        return;
    }

    try {
        setLoading(true);
        setSummary("");

        // Step 1: Fetch Summary
        const response = await axios.post("http://localhost:5000/summarize", {
            text,
            summary_level: Number(summaryLevel) / 100,
            summaryType
            
        }, {
            headers: { "Content-Type": "application/json" },
            withCredentials: true
        });
        if (response.data.error) {
            toast.error(response.data.error);
            return;
        }

        const generatedSummary = response.data.summary;
        setSummary(generatedSummary);
        console.log(response.data.accuracy)

        const saveResponse = await fetch("http://localhost:5000/save_history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
              text: text.trim(),
              summary,
              email: email.trim(),
          }),
        });
        
        const saveData = await saveResponse.json();
        if (saveResponse.ok) {
            toast.success("Summary saved successfully!");
        } else {
            alert("Error: " + saveData.error);
        }      
    } catch (error) {
        console.error("Error:", error);
        toast.error("Something went wrong. Please try again.");
    } finally {
        setLoading(false);
    }
  };


  const handleProcessFileSumm = async () => {
    if (!file) {
      alert("Please upload a file (.txt or .pdf) to summarize.");
      return;
    }
    try {
      setLoading(true);
      setSummary("");
      const formData = new FormData();
      formData.append("file", file);
      formData.append("summary_level", Number(summaryLevel)/100);
      formData.append("summaryType", summaryType);
      const response = await axios.post("http://localhost:5000/summarize_file", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      if (response.data.error) {
        alert(response.data.error);
        setSummary("");
      } else {
        setSummary(response.data.summary);
        setText(response.data.text)
      }
      const saveResponse = await fetch("http://localhost:5000/save_history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            text: text.trim(),
            summary,
            email: email.trim(),
        }),
      });
      
      const saveData = await saveResponse.json();
      if (saveResponse.ok) {
          toast.success("Summary saved successfully!");
      } else {
          alert("Error: " + saveData.error);
      }   

   // setSummary(response.data.summary || response.data.error);
    } catch (error) {
      console.error("Error summarizing file:", error);
      alert("Failed to fetch file summary. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProcessUrlSumm = async () => {
    if (!url.trim()) {
      alert("Please enter a Wikipedia URL.");
      return;
    }
    try {
      setLoading(true);
      setSummary("");
      const response = await axios.post("http://localhost:5000/summarize_url", {
        url,
        summary_level: Number(summaryLevel),
      }, { headers: { "Content-Type": "application/json" } });
      setSummary(response.data.summary || response.data.error);
    } catch (error) {
      console.error("Error summarizing URL:", error);
      alert("Failed to fetch Wikipedia summary. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProcessAudioSumm = async () => {
    if (!file) {
      alert("Please upload an audio file (.mp3) to summarize.");
      return;
    }
    try {
      setLoading(true);
      setSummary('');
      const formData = new FormData();
      formData.append("file", file);
      formData.append("summary_level", Number(summaryLevel) / 100);
      const response = await axios.post('http://localhost:5000/audio_summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSummary(response.data.summary || response.data.error);
      console.log(summary)
    } catch (error) {
      console.error("Error summarizing audio:", error);
      alert("Failed to fetch audio summary. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProcessTextTranscribe = async () => {
    if (!file) {
      alert("Please upload an audio file (.mp3) to transcribe.");
      return;
    }
    try {
      setLoading(true);
      setSummary('');
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post('http://localhost:5000/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSummary(response.data.transcript || response.data.error);
      console.log(summary)
    } catch (error) {
      console.error("Error transcribing audio:", error);
      alert("Failed to transcribe audio. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProcessTextTranslation = async () => {
    if (!text.trim()) {
        alert("Please enter some text to translate.");
        return;
    }
    try {
        setLoading(true);
        setSummary('');
        const response = await axios.post('http://localhost:5000/translate', {
            text,
            language: selectedLanguage,
        }, { headers: { 'Content-Type': 'application/json' } });

        const translatedText = response.data.translated_text || response.data.error;
        setSummary(translatedText);  
        
        const saveResponse = await fetch("http://localhost:5000/save_history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text.trim(),
                summary: translatedText,
                email: email.trim(),
            }),
        });

        const saveData = await saveResponse.json();
        if (saveResponse.ok) {
            toast.success("Summary saved successfully!");
        } else {
            alert("Error: " + saveData.error);
        }
    } catch (error) {
        console.error("Error translating:", error);
        alert("Failed to fetch translation. Please try again.");
    } finally {
        setLoading(false);
    }
  };



  const handleProcess = () => {
    if (mode === "text") handleProcessTextSumm();
    else if (mode === "audio") handleProcessAudioSumm();
    else if (mode === "file" || mode === "pdf") handleProcessFileSumm();
    else if (mode === "conversion") handleProcessTextTranscribe();
    else if (mode === "translation") handleProcessTextTranslation();
    else if (mode === "url") handleProcessUrlSumm();
  };

  const formatOutput = (text) => {
    if (!text) return null;
    switch (outputFormat) {
      case "bullet":
        return (
          <ul>
            {text.split('. ').map((sentence, index) => (
              sentence && <li key={index}>{sentence}</li>
            ))}
          </ul>
        );
      /*case "keypoints":
        const points = text.split('. ').filter((_, i) => i % 2 === 0); // Example: take every other sentence
        return (
          <ul>
            {points.map((point, index) => (
              point && <li key={index}>{point}</li>
            ))}
          </ul>
        );*/
      case "paragraph":
      default:
        return <p>{text}</p>;
    }
  };

  const toggleSidebar = async () => {
    setIsVisible(!isVisible);

    // Fetch data only when opening the sidebar
    if (!isVisible) {
        try {
            const response = await fetch(`http://localhost:5000/get_history?email=${encodeURIComponent(email)}`);
            const data = await response.json();
            if (response.ok) {
                setHistory(data);
                console.log("Email: ", email, history);
                
            } else {
                console.error("Error:", data.error);
            }
        } catch (err) {
            console.error("Fetch error:", err);
        }
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
        toast.success("Copied to clipboard!");
    }).catch(err => {
        toast
        .error("Failed to copy:", err);
    }); 
  };

  const readTextAloud = (text) => {
    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = "en-US";
    speech.rate = 1.0; 
    speech.pitch = 1.0; 
    window.speechSynthesis.speak(speech);
  };

  const stopSpeech = () => {
    window.speechSynthesis.cancel();
  };

  return (
    <div className="container">
      <div className="input-section">
        <strong>Select Input Method:</strong>
        <div className="menu">
          {["text", "file", "pdf"].map((option) => (
            <label key={option} className={`menu-item1 ${inputMethod === option ? "selected" : ""}`}>
              <input
                type="radio"
                value={option}
                checked={inputMethod === option}
                onChange={() => setInputMethod(option)}
                disabled={mode === "audio" || mode === "conversion"}
              />
              <div className="input-methods-container">
                {option === "text" ? (
                  <>
                    <ContentCopyIcon /> <p className="input-method-text">Raw Text</p>
                  </>
                ) : option === "file" ? (
                  <>
                    <FilePresentIcon /> <p className="input-method-text">Load File</p>
                  </>
                ) : (
                  <>
                    <InsertDriveFileIcon /> <p className="input-method-text">Load .pdf</p>
                  </>
                )}
              </div>
            </label>
          ))}
        </div>

        {inputMethod === "text" && mode !== "audio" && mode !== "conversion" && (
          <textarea
            className="textArea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text here..."
          />
        )}
        {(inputMethod === "file" || inputMethod === "pdf" || mode === "audio" || mode === "conversion") && (
          <input
            type="file"
            onChange={handleFileUpload}
            accept={mode === "audio" || mode === "conversion" ? ".mp3" : inputMethod === "file" ? ".txt" : ".pdf"}
          />
        )}

        <strong className='process-type-heading'>Processing Type:</strong>
        <div className="menu">
          {["text", "file", "audio", "conversion", "translation"].map((option) => (
            <label key={option} className="menu-item">
              <input
                type="radio"
                value={option}
                checked={mode === option}
                onChange={() => setMode(option)}
              />
              {option === "text" ? "Text Summarization" :
               option === "file" ? "File Summarization" :
               option === "audio" ? "Audio Summarization (.mp3)" :
               option === "conversion" ? "Convert Audio (.mp3) to Text" :
               "Text Translation (.txt/.pdf)"}
            </label>
          ))}
        </div>

        {mode === "translation" && (
          <>
            <strong>Select Language:</strong>
            <select value={selectedLanguage} onChange={(e) => setSelectedLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="it">Italian</option>
              <option value="ja">Japanese</option>
              <option value="ta">Tamil</option>
            </select>
          </>
        )}

        {mode !== 'translation' && (
          <div className='summ-type-container'>
            <div className="menu">
              {[25, 50, 75].map((level) => (
                <label key={level} className="menu-item">
                  <input
                    type="radio"
                    value={level}
                    checked={summaryLevel === level}
                    onChange={() => setSummaryLevel(level)}
                  />
                  {level}%
                </label>
              ))}
            </div>
            <div className="menu">
              <label>
                <input type="radio" value={summaryType}
                    checked={summaryType === "abstractive"}
                    onChange={() => setSummaryType("abstractive")}/>
                Abstractive Summarization
              </label>
              <label>
                <input type="radio" value={summaryType}
                    checked={summaryType === "extractive"}
                    onChange={() => setSummaryType("extractive")}/>
                Extractive Summarization
              </label>
            </div>
          </div>
        )}

        <button className="button" onClick={handleProcess} disabled={loading}>
          {loading ? "Processing..." : "Process"}
        </button>
      </div>

      <div className="output-section">
        <div className="charCount-container">
          {inputMethod=="text" ?
          <>
            <p className="charCount">Input Length: {text.length || (file ? "Loading..." : 0)} char</p>
            <p className="charCount">Summary Length: {summary.length} char</p>
          </> : <></>
          }
          <div className="icon-container">
            {!loading && true && (
              <button
                className="copy-btn"
                onClick={() => {
                  navigator.clipboard.writeText(summary);
                  toast.success("Summary copied to clipboard!");
                }}
              >
                <ContentCopyIcon />
              </button>
            )}
            {!loading && true && (
              <button
                className="download-btn"
                onClick={() => {
                  const blob = new Blob([summary], { type: "text/plain" });
                  const link = document.createElement("a");
                  link.href = URL.createObjectURL(blob);
                  link.download = "summary.txt";
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                }}
              >
                <DownloadIcon />
              </button>
            )}
            <button className="history-btn" onClick={toggleSidebar}>
                <HistoryIcon />
            </button>
            <button className="history-btn" onClick={() => setIsAuthenticated(false)}>
                <LogoutIcon />
            </button>

            <div className={`${isVisible ? "show history-sidebar" : "history-sidebar-not-show"}`}>
                <button className="close-btn" onClick={toggleSidebar}>×</button>
                <h2>History</h2>
                <table className="history-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Input Text</th>
                            <th>Output Summary</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history.map((item) => (
                            <tr key={item.id}>
                                <td>{item.created_at}</td>
                                <td className="clickable" onClick={() => copyToClipboard(item.input_text)}>
                                    {item.input_text.slice(0, 50)}... <span className="copy-hint">[Copy]</span>
                                </td>
                                <td className="clickable" onClick={() => copyToClipboard(item.output_summ)}>
                                    {item.output_summ.slice(0, 50)}... <span className="copy-hint">[Copy]</span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
          </div>
        </div>

        <strong>Output Format:</strong>
        <div className="menu">
          {["paragraph", "bullet"].map((format) => (
            <label key={format} className="menu-item">
              <input
                type="radio"
                value={format}
                checked={outputFormat === format}
                onChange={() => setOutputFormat(format)}
              />
              {format.charAt(0).toUpperCase() + format.slice(1)}
            </label>
          ))}
          <div className="text-to-speech" onClick={() => readTextAloud(summary)}>
              <SpatialTrackingIcon />
          </div>
          <button className="stop-btn" onClick={stopSpeech}>Stop</button>
        </div>

        <div className="summary-output">
          <strong>Output:</strong>
          {loading ? <p>Loading...</p> : formatOutput(summary)}
        </div>
      </div>
    </div>
  );
}

export default Home;