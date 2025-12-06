import axios from 'axios';

// Create an instance of axios with the base URL
const api = axios.create({
    baseURL: "https://hvzl4zsm-8000.use2.devtunnels.ms/"
})

export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    
    const response = await api.post("/upload", formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });
    return response.data;
};

// Export the Axios instance
export default api;
