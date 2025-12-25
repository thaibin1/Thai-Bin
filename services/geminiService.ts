
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { ImageAsset } from "../types";

// Manage Manual API Key
let manualApiKey: string | null = typeof window !== 'undefined' ? localStorage.getItem('gemini_api_key') : null;

export const setManualApiKey = (key: string) => {
  manualApiKey = key;
  if (typeof window !== 'undefined') {
    localStorage.setItem('gemini_api_key', key);
  }
};

export const clearManualApiKey = () => {
  manualApiKey = null;
  if (typeof window !== 'undefined') {
    localStorage.removeItem('gemini_api_key');
  }
};

const getApiKey = () => {
  return manualApiKey || process.env.API_KEY;
};

// Helper to remove data URL prefix for API
const cleanBase64 = (dataUrl: string) => {
  if (dataUrl.includes(',')) {
    return dataUrl.split(',')[1];
  }
  return dataUrl;
};

// --- Helpers ---

const retry = async <T>(fn: () => Promise<T>, maxRetries = 3, initialDelay = 2000): Promise<T> => {
  let lastError: any;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      const errorMsg = error.message || "";
      const isTransient = 
        errorMsg.includes('500') || 
        errorMsg.includes('503') || 
        errorMsg.toLowerCase().includes('overloaded') || 
        errorMsg.toLowerCase().includes('deadline expired') ||
        errorMsg.toLowerCase().includes('internal error');
      
      if (!isTransient || i === maxRetries - 1) {
        throw error;
      }
      
      const delay = initialDelay * Math.pow(2, i);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw lastError;
};

const extractImageFromResponse = (response: any): string => {
  const candidate = response.candidates?.[0];
  
  if (!candidate) {
    throw new Error("API không trả về kết quả nào. Kiểm tra kết nối mạng.");
  }

  const safetyRatings = candidate.safetyRatings || [];
  const blockedRating = safetyRatings.find((r: any) => r.blocked === true);

  if (candidate.finishReason === 'SAFETY' || blockedRating) {
    throw new Error("Hình ảnh bị chặn bởi bộ lọc an toàn. Vui lòng thử ảnh khác ít nhạy cảm hơn.");
  }

  const parts = candidate.content?.parts;
  if (!parts || parts.length === 0) {
    throw new Error("AI không tạo được ảnh. Vui lòng thử lại.");
  }
  
  for (const part of parts) {
    if (part.inlineData && part.inlineData.data) {
      return `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
    }
  }
  
  const textPart = parts.find((p: any) => p.text);
  if (textPart) {
    throw new Error(`Thông báo từ AI: ${textPart.text}`);
  }

  throw new Error("Không tìm thấy dữ liệu hình ảnh.");
};

const handleError = (error: any) => {
  console.error("Gemini API Error:", error);
  const errorMsg = error.message || "";
  if (errorMsg.includes('404') || errorMsg.includes('not found') || errorMsg.includes('API key')) {
       throw new Error("Lỗi API Key: Vui lòng kiểm tra lại key hoặc dự án.");
  }
  throw new Error(errorMsg || "Xử lý thất bại.");
};

// --- Main Services ---

const executeSingleTryOn = async (
  personImage: ImageAsset,
  garmentImage: ImageAsset | null,
  garmentDetailImage: ImageAsset | null,
  accessoryImage: ImageAsset | null,
  instructions: string,
  aspectRatio: string,
  imageSize: string,
  modelName: string,
  tryOnMode: string = 'keep-model-bg'
): Promise<string | null> => {
  const apiKey = getApiKey();
  if (!apiKey) return null;

  const ai = new GoogleGenAI({ apiKey });

  let modeSpecificPrompt = "";
  switch (tryOnMode) {
    case 'keep-model-bg':
      modeSpecificPrompt = `
        - GOAL: Dress the person in IMAGE 1 with the outfit from IMAGE 2.
        - PRIMARY SOURCE (Person & BG): IMAGE 1 is the template for body, face, pose, and background.
        - SECONDARY SOURCE (Garment): IMAGE 2 is the clothing to use.
        - TASK: Replace the existing clothing in IMAGE 1 with the clothing from IMAGE 2. 
        - CONSTRAINT: Keep 100% of the original background and person's identity from IMAGE 1.
      `;
      break;
    case 'new-bg':
      modeSpecificPrompt = `
        - GOAL: Professional studio fashion photography.
        - SOURCE 1: Use the person's face and build from IMAGE 1.
        - SOURCE 2: Use the clothing from IMAGE 2.
        - ENVIRONMENT: Replace the background with a clean, high-end, minimalist professional photography studio background with soft lighting.
      `;
      break;
    case 'keep-garment-bg':
      modeSpecificPrompt = `
        - GOAL: Face-Swap onto a model in a specific scene.
        - SOURCE 1 (Person): Use 100% of the facial features, skin tone, and identity from IMAGE 1.
        - SOURCE 2 (Scene): IMAGE 2 is the MASTER for background, clothing, and environment.
        - TASK: Take the person's head from IMAGE 1 and place it onto the body/model in IMAGE 2.
        - CONSTRAINT: Retain 100% of the original clothing, outfit, and background from IMAGE 2.
      `;
      break;
  }

  const basePrompt = `
    You are a professional digital fashion editor. 
    ${modeSpecificPrompt}
    
    GUIDELINES:
    1. Result must be highly photorealistic.
    2. Seamless blending of skin tones.
    3. Realistic fabric rendering.
    4. Aspect Ratio: ${aspectRatio}.
    ${instructions ? `5. Additional modification: ${instructions}` : ''}
    
    OUTPUT: Provide only the final edited image.
  `;

  const parts: any[] = [
    { inlineData: { mimeType: personImage.mimeType, data: cleanBase64(personImage.data) } } // IMAGE 1
  ];

  if (garmentImage) {
    parts.push({ inlineData: { mimeType: garmentImage.mimeType, data: cleanBase64(garmentImage.data) } }); // IMAGE 2
  }

  if (garmentDetailImage) {
    parts.push({ inlineData: { mimeType: garmentDetailImage.mimeType, data: cleanBase64(garmentDetailImage.data) } });
  }

  if (accessoryImage) {
    parts.push({ inlineData: { mimeType: accessoryImage.mimeType, data: cleanBase64(accessoryImage.data) } });
  }

  parts.push({ text: basePrompt });

  try {
    const config: any = { 
      imageConfig: { aspectRatio },
      safetySettings: [
        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
      ]
    };
    
    if (modelName === 'gemini-3-pro-image-preview') {
        config.imageConfig.imageSize = imageSize;
    }

    const response = await retry<GenerateContentResponse>(() => ai.models.generateContent({
      model: modelName,
      contents: { parts: parts },
      config: config
    }));
    
    return extractImageFromResponse(response);
  } catch (error: any) {
    throw error;
  }
};

export const generateVirtualTryOn = async (
  personImage: ImageAsset,
  garmentImage: ImageAsset | null,
  garmentDetailImage: ImageAsset | null,
  accessoryImage: ImageAsset | null,
  instructions: string,
  aspectRatio: string = "9:16",
  imageSize: string = "2K",
  modelName: string = "gemini-3-pro-image-preview",
  tryOnMode: string = 'keep-model-bg',
  count: number = 1
): Promise<string[]> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");

  const tasks = [];
  for (let i = 0; i < count; i++) {
    tasks.push(executeSingleTryOn(
      personImage, 
      garmentImage, 
      garmentDetailImage, 
      accessoryImage, 
      instructions, 
      aspectRatio, 
      imageSize, 
      modelName, 
      tryOnMode
    ));
  }
  
  const results = await Promise.allSettled(tasks);
  const validResults: string[] = [];
  let lastErrorMsg = "";

  results.forEach((res) => {
    if (res.status === 'fulfilled' && res.value) {
      validResults.push(res.value);
    } else if (res.status === 'rejected') {
      lastErrorMsg = res.reason?.message || "Lỗi hệ thống.";
    }
  });

  if (validResults.length === 0) {
    throw new Error(lastErrorMsg || "Không thể tạo ảnh.");
  }

  return validResults;
};

export const changeImageBackground = async (
  imageDataUrl: string,
  backgroundPrompt: string = "A modern studio background.",
  detailImage: ImageAsset | null = null,
  aspectRatio: string = "9:16",
  imageSize: string = "2K",
  modelName: string = "gemini-3-pro-image-preview",
  customBgImage: ImageAsset | null = null
): Promise<string> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");
  const ai = new GoogleGenAI({ apiKey });

  const prompt = `Keep the subject identical. Replace background with: ${customBgImage ? 'the environment from the reference' : backgroundPrompt}.`;
  const parts: any[] = [{ inlineData: { mimeType: 'image/png', data: cleanBase64(imageDataUrl) } }];
  if (detailImage) parts.push({ inlineData: { mimeType: detailImage.mimeType, data: cleanBase64(detailImage.data) } });
  if (customBgImage) parts.push({ inlineData: { mimeType: customBgImage.mimeType, data: cleanBase64(customBgImage.data) } });
  parts.push({ text: prompt });

  try {
    const config: any = { 
      imageConfig: { aspectRatio },
      safetySettings: [
        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
      ]
    };
    if (modelName === 'gemini-3-pro-image-preview') config.imageConfig.imageSize = imageSize;
    const response = await ai.models.generateContent({ model: modelName, contents: { parts }, config });
    return extractImageFromResponse(response);
  } catch (error) { handleError(error); return ""; }
};

export const changeImageBackgroundBatch = async (
  imageDataUrl: string,
  prompts: string[],
  detailImage: ImageAsset | null = null,
  aspectRatio: string = "9:16",
  imageSize: string = "2K",
  modelName: string = "gemini-3-pro-image-preview",
  customBgImage: ImageAsset | null = null
): Promise<string[]> => {
  const results = await Promise.all(prompts.map(p => changeImageBackground(imageDataUrl, p, detailImage, aspectRatio, imageSize, modelName, customBgImage).catch(() => null)));
  return results.filter((r): r is string => r !== null);
};

export const analyzeOutfit = async (image: ImageAsset, detailImage: ImageAsset | null): Promise<string> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");
  const ai = new GoogleGenAI({ apiKey });
  const parts: any[] = [{ inlineData: { mimeType: image.mimeType, data: cleanBase64(image.data) } }];
  if (detailImage) parts.push({ inlineData: { mimeType: detailImage.mimeType, data: cleanBase64(detailImage.data) } });
  parts.push({ text: "Describe this outfit for high-quality video generation." });
  const response = await ai.models.generateContent({ model: 'gemini-3-pro-preview', contents: { parts } });
  return response.text || "";
};

export const generatePromptsFromAnalysis = async (analysis: string, count: number): Promise<string[]> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");
  const ai = new GoogleGenAI({ apiKey });
  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: analysis,
    config: {
      systemInstruction: `Generate ${count} video motion prompts. Return JSON { "prompts": [] }.`,
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: { prompts: { type: Type.ARRAY, items: { type: Type.STRING } } }
      }
    }
  });
  const parsed = JSON.parse(response.text || "{}");
  return parsed.prompts || [];
};
