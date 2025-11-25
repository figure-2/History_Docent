'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { HISTORICAL_PERSONAS, LANGUAGES } from '@/lib/constants';
import type { Persona, Language } from '@/lib/types';
import { Languages, Users } from 'lucide-react';
import Image from 'next/image';
import { Logo } from '../icons/logo';

interface HeaderProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
  persona: Persona;
  onPersonaChange: (persona: Persona) => void;
}

export function AppHeader({
  language,
  onLanguageChange,
  persona,
  onPersonaChange,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 w-full border-b bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto flex h-20 items-center justify-between px-4 md:px-6">
        <div className="flex items-center gap-3">
          <Logo className="h-8 w-8 text-primary" />
          <h1 className="font-headline text-2xl font-bold tracking-tight text-foreground">
            한양에 왔습니다~
          </h1>
        </div>
        <div className="flex items-center gap-2 md:gap-4">
          <div className="flex items-center gap-2">
            <Languages className="h-4 w-4 text-muted-foreground" />
            <Select
              value={language.code}
              onValueChange={(code) => {
                const newLang = LANGUAGES.find((l) => l.code === code);
                if (newLang) {
                  onLanguageChange(newLang);
                }
              }}
            >
              <SelectTrigger className="w-28 border-0 bg-transparent shadow-none focus:ring-0 md:w-32">
                <SelectValue placeholder="Select Language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <Select
              value={persona.name[language.code]}
              onValueChange={(name) => {
                const newPersona = HISTORICAL_PERSONAS.find(
                  (p) => p.name[language.code] === name
                );
                if (newPersona) {
                  onPersonaChange(newPersona);
                }
              }}
            >
              <SelectTrigger className="w-36 border-0 bg-transparent shadow-none focus:ring-0 md:w-40">
                <SelectValue placeholder="Select Persona" />
              </SelectTrigger>
              <SelectContent>
                {HISTORICAL_PERSONAS.map((p) => (
                  <SelectItem key={p.avatarId} value={p.name[language.code]}>
                    {p.name[language.code]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </header>
  );
}
