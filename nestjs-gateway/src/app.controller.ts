import { Controller, Post, Body, UseInterceptors, UploadedFile, HttpException, HttpStatus, HttpCode } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { AppService } from './app.service';

@Controller('api/rag')
export class AppController {
  constructor(private readonly appService: AppService) {}

  /**
   * 💬 Chat Interface Endpoint
   * Accepts: { "question": "What's a Bitcoin?" }
   * Returns: { "answer": "...", "groundingScore": 0.95, "status": "VERIFIED_BY_LEXICAL_MATCH" }
   */
  @Post('ask')
  @HttpCode(HttpStatus.OK)
  async askQuestion(@Body('question') question: string) {
    if (!question || question.trim() === '') {
      throw new HttpException('Question input cannot be blank', HttpStatus.BAD_REQUEST);
    }
    return this.appService.handleUserQuery(question);
  }

  /**
   * 🚀 File Ingestion & Indexing Endpoint
   * Accepts: Multipart FormData containing 'file' key
   * Returns: { "status": "SUCCESS", "totalChunks": 12, "collection": "user_dynamic_workspace" }
   */
  @Post('upload')
  @HttpCode(HttpStatus.OK)
  @UseInterceptors(FileInterceptor('file'))
  async uploadUserDocument(
    @UploadedFile() file: { buffer: Buffer; originalname: string; mimetype: string } // ✨ FIXED: Patched directly to clear namespace error
  ) { 
    if (!file) {
      throw new HttpException('No file stream detected in request body metadata', HttpStatus.BAD_REQUEST);
    }
    return this.appService.processAndIndexDocument(file);
  }
}