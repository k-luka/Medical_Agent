import axios from 'axios';

// Create an instance of axios with the base URL
const api = axios.create({
    baseURL: "https://hvzl4zsm-8000.use2.devtunnels.ms/"
})

// Export the Axios instance
export default api;