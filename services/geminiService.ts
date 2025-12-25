
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
  const parts = response.candidates?.[0]?.content?.parts;
  if (!parts) throw new Error("Không có nội dung nào được tạo ra.");
  for (const part of parts) {
    if (part.inlineData && part.inlineData.data) {
      return `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
    }
  }
  throw new Error("Không tìm thấy dữ liệu hình ảnh trong phản hồi.");
};

const handleError = (error: any) => {
  console.error("Gemini API Error:", error);
  const errorMsg = error.message || "";
  if (errorMsg.includes('404') || errorMsg.includes('not found') || errorMsg.includes('API key')) {
       throw new Error("Lỗi xác thực: Key không hợp lệ hoặc dự án chưa liên kết thanh toán.");
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
        - **MANDATORY CANVAS**: Use **IMAGE 1** (Person Upload) as the base layer.
        - **ZERO BACKGROUND CHANGE**: You MUST NOT change any pixels in the background or surroundings of **IMAGE 1**. 
        - **SWAP TARGET**: Only modify the clothing area of the person in **IMAGE 1**. Replace the existing clothes with the garment from **IMAGE 2**.
        - **CONSISTENCY**: Keep the original model's pose, hair, skin tone, and environment exactly as they appear in **IMAGE 1**.
      `;
      break;
    case 'new-bg':
      modeSpecificPrompt = `
        - **IDENTITY**: Take the person's face and body shape from **IMAGE 1**.
        - **OUTFIT**: Dress them in the garment from **IMAGE 2**.
        - **NEW SCENE**: Place them in a professional high-fashion studio background with soft bokeh.
      `;
      break;
    case 'keep-garment-bg':
      modeSpecificPrompt = `
        - **MANDATORY CANVAS**: Use **IMAGE 2** (Garment/Model) as the base layer.
        - **IDENTITY TRANSFER**: Keep the background and the clothes in **IMAGE 2** exactly as they are. 
        - **FACE SWAP**: Only replace the face of the person in **IMAGE 2** with the face of the person from **IMAGE 1**.
      `;
      break;
  }

  const basePrompt = `
    Role: Professional AI Fashion Retoucher.
    
    ${modeSpecificPrompt}
    
    Technical Details:
    - Result must be photorealistic, 100% natural fabric folds and lighting.
    - Give the person a very gentle, friendly smile (mỉm cười nhẹ nhàng).
    - Resolution: ${imageSize}.
    ${instructions ? `- User note: ${instructions}` : ''}
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
    const config: any = { imageConfig: { aspectRatio: aspectRatio } };
    if (modelName === 'gemini-3-pro-image-preview') config.imageConfig.imageSize = imageSize;

    const response = await retry<GenerateContentResponse>(() => ai.models.generateContent({
      model: modelName,
      contents: { parts: parts },
      config: config
    }));
    
    return extractImageFromResponse(response);
  } catch (error: any) {
    console.warn("Try-on generation failed:", error);
    return null;
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
  count: number = 2 // New parameter to control batch count
): Promise<string[]> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");

  // Create an array of tasks based on the count requested
  const tasks = Array.from({ length: count }).map(() => 
    executeSingleTryOn(personImage, garmentImage, garmentDetailImage, accessoryImage, instructions, aspectRatio, imageSize, modelName, tryOnMode)
  );
  
  const results = await Promise.all(tasks);
  const validResults = results.filter((r): r is string => r !== null);

  if (validResults.length === 0) throw new Error("Google AI không thể tạo ảnh. Vui lòng thử lại.");
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

  const prompt = `
    Replace background of image.
    Keep subject identical. Natural gentle smile.
    ${customBgImage ? 'Use custom background provided.' : `Theme: ${backgroundPrompt}`}
  `;

  const parts: any[] = [{ inlineData: { mimeType: 'image/png', data: cleanBase64(imageDataUrl) } }];
  if (detailImage) parts.push({ inlineData: { mimeType: detailImage.mimeType, data: cleanBase64(detailImage.data) } });
  if (customBgImage) parts.push({ inlineData: { mimeType: customBgImage.mimeType, data: cleanBase64(customBgImage.data) } });
  parts.push({ text: prompt });

  try {
    const config: any = { imageConfig: { aspectRatio } };
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

export const changeImageBackgroundAndPoseBatch = async (
  imageDataUrl: string,
  configs: { background: string; pose: string }[],
  detailImage: ImageAsset | null = null,
  aspectRatio: string = "9:16",
  imageSize: string = "2K",
  modelName: string = "gemini-3-pro-image-preview",
  customBgImage: ImageAsset | null = null
): Promise<string[]> => [];

export const changeImagePose = async (
  imageDataUrl: string,
  posePrompt: string,
  modelName: string = "gemini-3-pro-image-preview"
): Promise<string> => "";

export const analyzeOutfit = async (image: ImageAsset, detailImage: ImageAsset | null): Promise<string> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("Thiếu API Key.");
  const ai = new GoogleGenAI({ apiKey });
  const parts: any[] = [{ inlineData: { mimeType: image.mimeType, data: cleanBase64(image.data) } }];
  if (detailImage) parts.push({ inlineData: { mimeType: detailImage.mimeType, data: cleanBase64(detailImage.data) } });
  parts.push({ text: "Analyze person and outfit details." });

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
      systemInstruction: `Generate ${count} video prompts. JSON format.`,
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
