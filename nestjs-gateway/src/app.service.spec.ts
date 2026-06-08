import { Test, TestingModule } from '@nestjs/testing';
import { AppService } from './app.service';
import { ConfigService } from '@nestjs/config';

describe('AppService (Unit Tests)', () => {
  let service: AppService;

  beforeEach(async () => {
    const mockConfigService = {
      get: jest.fn((key: string) => {
        if (key === 'PYTHON_SERVICE_BASE_URL') return 'http://127.0.0.1:5000';
        return null;
      }),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AppService,
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<AppService>(AppService);
  });

  describe('🧠 Text Chunking Engine', () => {
    it('should cleanly split a long string along sentence boundaries', () => {
      const text = 'First technical sentence. Second valid statement. Third operational log.';
      const chunks = service['chunkText'](text, 30, 5);

      expect(chunks.length).toBeGreaterThanOrEqual(1);
      expect(chunks[0]).toContain('First technical sentence.');
    });
  });

  describe('📊 Lexical Grounding Verification Matcher', () => {
    it('should output a high confidence metric score for exact keyword-matching contexts', () => {
      const context = 'The failover engine triggers automatic core replication within 45 seconds.';
      const answer = 'The core replication triggers within 45 seconds.';

      const score = service['calculateGroundingScore'](answer, context);
      
      expect(score).toBeGreaterThanOrEqual(0.3);
      expect(typeof score).toBe('number');
    });

    it('should drop the confidence metric to zero if vocabulary has zero matching tokens', () => {
      const context = 'The database cluster is reading records normally.';
      const answer = 'The space shuttle launched from Florida yesterday.';

      const score = service['calculateGroundingScore'](answer, context);
      
      expect(score).toBe(0);
    });
  });
});