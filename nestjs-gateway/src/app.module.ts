import { Module } from '@nestjs/common';
import { ServeStaticModule } from '@nestjs/serve-static';
import { ConfigModule } from '@nestjs/config'; // ◄── Already imported! Keep this line.
import { join } from 'path';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    // 🚀 Register the configuration framework globally across the application ecosystem
    ConfigModule.forRoot({ 
      isGlobal: true 
    }),

    // Serves our HTML interface directly from the public directory
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, '..', 'public'),
    }),
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}