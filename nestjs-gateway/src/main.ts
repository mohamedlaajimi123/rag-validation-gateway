import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { GlobalExceptionFilter } from './http-exception.filter'; // 🛡️ IMPORT FILTER
import { Logger } from '@nestjs/common';
import * as express from 'express';

async function bootstrap() {
  const logger = new Logger('Bootstrap');
  const app = await NestFactory.create(AppModule);

  // 🛡️ BIND GLOBAL EXCEPTION FILTER: Prevents stack traces from leaking to clients
  app.useGlobalFilters(new GlobalExceptionFilter());

  // 🔓 Expand global payload limits to handle PDF file buffers safely
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));

  // 🔒 HARDENED CORS: Replace wide-open rules with a structured environment policy
  app.enableCors({
    origin: process.env.ALLOWED_ORIGINS 
      ? process.env.ALLOWED_ORIGINS.split(',') 
      : 'http://localhost:3000', // Points explicitly to your frontend dev port
    methods: 'GET,POST',
    credentials: true,
  });

  const port = 3001;
  await app.listen(port);
  
  // 📡 Telemetry: Replaced standard console.log with NestJS structural logger
  logger.log(`🚀 NestJS RAG Gateway is listening securely on http://localhost:${port}`);
}
bootstrap();