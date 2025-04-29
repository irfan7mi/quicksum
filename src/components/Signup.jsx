import React, { useState } from "react";
import "./Auth.css"; 
import axios from "axios";
import { toast } from "react-toastify";

const Signup = ({ setAuthType }) => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    mobile: "",
    password: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/signup", formData, {
        headers: { "Content-Type": "application/json" },
        withCredentials: true, // ✅ Allow credentials (important for CORS)
      });
      toast.success(response.data.message)
    } catch (error) {
      toast.error("Signup failed");
    }
  };


  return (
    <div className="auth-container">
      <h2>Sign Up</h2>
      <form onSubmit={handleSubmit}>
        <input type="text" name="username" placeholder="Username" onChange={handleChange} required />
        <input type="email" name="email" placeholder="Email" onChange={handleChange} required />
        <input type="text" name="mobile" placeholder="Mobile No" onChange={handleChange} required />
        <input type="password" name="password" placeholder="Password" onChange={handleChange} required />
        <button type="submit">Register</button>
        <p className="switch-auth" onClick={() => setAuthType("login")}>Already have an account? Login</p>
      </form>
    </div>
  );
};

export default Signup;
