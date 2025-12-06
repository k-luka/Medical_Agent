import React, { useState, useEffect, useRef } from "react";
import api, { uploadFile } from "../api";
import { Send, Bot, User, Loader2, Cpu, UploadCloud, RotateCcw } from "lucide-react";
import ReactMarkdown from 'react-markdown';

const Chat = () => {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hello. I am your Medical Imaging Agent. Drop a CT scan file here to begin." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false); // Global loading state (for upload/reset)
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // --- Core Chat Logic (Streaming) ---
  const processMessage = async (content, role = "user") => {
    // 1. Add User Message
    const newMsg = { role, content };
    setMessages((prev) => [...prev, newMsg]);
    
    if (role === "user") {
      // Add a temporary "Status" bubble
      const statusId = "temp-status-" + Date.now();
      setMessages(prev => [...prev, { role: "status", content: "Thinking...", id: statusId }]);

      try {
        // --- FIX 1: Sanitize URL to prevent double slashes ---
        const baseUrl = api.defaults.baseURL.replace(/\/$/, "");
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: content })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let finalReply = "";
        let visualFile = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          // Split by newline because backend sends NDJSON
          const lines = chunk.split("\n").filter(line => line.trim() !== "");

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              
              if (data.type === "status") {
                // Update the status bubble text (e.g., "Running tool...")
                setMessages(prev => prev.map(m => 
                  m.id === statusId ? { ...m, content: data.text } : m
                ));
              } 
              else if (data.type === "reply") {
                finalReply = data.text;
                visualFile = data.visual_file;
              }
            } catch (e) {
              console.error("Error parsing JSON chunk", e);
            }
          }
        }

        // Replace Status bubble with Final Assistant Reply
        setMessages(prev => {
           const filtered = prev.filter(m => m.id !== statusId);
           const newMsg = { role: "assistant", content: finalReply };
           if (visualFile) {
             const baseUrl = api.defaults.baseURL.replace(/\/$/, "");
             newMsg.imageSrc = `${baseUrl}/image/${visualFile}`;
           }
           return [...filtered, newMsg];
        });

      } catch (err) {
        console.error(err);
        setMessages(prev => prev.filter(m => m.id !== statusId));
        setMessages(prev => [...prev, { role: "assistant", content: "Error: Connection failed." }]);
      }
    }
  };

  const handleSend = () => {
    if (!input.trim()) return;
    processMessage(input);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // --- Drag & Drop Handlers ---
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files?.length > 0) {
      await handleFileUpload(files[0]);
    }
  };

  // --- File Upload Logic ---
  const handleFileUpload = async (file) => {
    setLoading(true); // Show global spinner only for the upload part

    try {
      // 1. Send file to backend
      const response = await uploadFile(file);
      setLoading(false); // Stop spinner before starting chat
      
      // 2. Trigger Agent Awareness via Chat
      // We use processMessage so it streams the response properly
      const hiddenPrompt = `I have uploaded a file named "${response.filename}". Please acknowledge it and tell me what you know about it.`;
      
      // This will add a user bubble and then stream the agent's reply
      await processMessage(hiddenPrompt);

    } catch (err) {
      setLoading(false);
      setMessages((prev) => [...prev, { role: "assistant", content: "❌ Error: File upload failed." }]);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      await api.post("/reset");
      setMessages([
        { role: "assistant", content: "Hello. I am your Medical Imaging Agent. Drop a CT scan file here to begin." }
      ]);
      setInput("");
    } catch (err) {
      console.error("Reset failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div 
      className="flex flex-col h-screen bg-slate-50 text-slate-900 font-sans relative"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      
      {/* --- Drag Overlay --- */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-blue-600/10 backdrop-blur-sm border-4 border-blue-500 border-dashed m-4 rounded-xl flex items-center justify-center pointer-events-none">
          <div className="bg-white p-8 rounded-2xl shadow-xl flex flex-col items-center animate-bounce">
            <UploadCloud size={48} className="text-blue-600 mb-4" />
            <h3 className="text-xl font-bold text-slate-800">Drop CT Scan Here</h3>
            <p className="text-slate-500">I will upload it to the secure sandbox.</p>
          </div>
        </div>
      )}

      {/* --- Header --- */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center shadow-sm sticky top-0 z-10">
        <div className="p-2 bg-blue-600 rounded-lg mr-3">
          <Cpu className="text-white w-6 h-6" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-slate-800">Medical AI Agent</h1>
          <p className="text-xs text-slate-500 flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-1.5 inline-block"></span>
            Online • Powered by Gemini & MONAI
          </p>
        </div>

        <button 
          onClick={handleReset}
          className="ml-auto flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors border border-slate-200 hover:border-blue-200"
          title="Start New Chat"
        >
          <RotateCcw size={16} />
          <span className="hidden sm:inline">New Chat</span>
        </button>
      </header>

      {/* --- Chat Area --- */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((m, idx) => (
            <div key={idx} className={`flex w-full ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              
              {m.role === "status" ? (
                <div className="flex w-full justify-start animate-pulse">
                  <div className="flex items-center ml-14 space-x-2 text-slate-500 text-sm italic">
                    <Loader2 size={14} className="animate-spin" />
                    <span>{m.content}</span>
                  </div>
                </div>
              ) : (
                <div className={`flex max-w-[85%] md:max-w-[75%] ${m.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                  
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mx-2 ${
                    m.role === "user" ? "bg-blue-600" : "bg-emerald-600"
                  }`}>
                    {m.role === "user" ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
                  </div>

                  <div className={`p-4 rounded-2xl shadow-sm text-sm leading-relaxed ${
                      m.role === "user"
                        ? "bg-blue-600 text-white rounded-tr-none whitespace-pre-wrap"
                        : "bg-white border border-slate-200 text-slate-800 rounded-tl-none"
                    }`}>
                    {m.role === "user" ? m.content : (
                      <ReactMarkdown
                        components={{
                          ul: ({ node, ...props }) => <ul className="list-disc pl-5 space-y-1" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal pl-5 space-y-1" {...props} />,
                          strong: ({ node, ...props }) => <span className="font-bold text-blue-800" {...props} />,
                          p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />
                        }}
                      >
                        {m.content}
                      </ReactMarkdown>
                    )}
                    {m.imageSrc && (
                      <div className="mt-3">
                        <img
                          src={m.imageSrc}
                          alt="Agent visual"
                          className="rounded-xl border border-slate-200 shadow-sm max-h-72 object-contain"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {/* Global Loader (Only for uploads/reset now) */}
          {loading && (
            <div className="flex justify-start w-full">
              <div className="flex flex-row items-center ml-2">
                <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center mx-2">
                  <Loader2 size={16} className="text-white animate-spin" />
                </div>
                <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-tl-none shadow-sm text-sm text-slate-500 italic">
                  Uploading...
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* --- Input Area --- */}
      <div className="bg-white border-t border-slate-200 p-4">
        <div className="max-w-4xl mx-auto relative flex items-center">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message or drop a file..."
            className="w-full bg-slate-100 text-slate-900 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all resize-none shadow-inner text-sm"
            rows={1}
            style={{ minHeight: "50px", maxHeight: "150px" }}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="absolute right-2 p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;