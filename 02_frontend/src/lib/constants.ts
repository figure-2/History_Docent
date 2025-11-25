import type { Persona, Language } from '@/lib/types';
import { PlaceHolderImages } from '@/lib/placeholder-images';

export const LANGUAGES: Language[] = [
  { name: '한국어', code: 'ko' },
  { name: 'English', code: 'en' },
  { name: '日本語', code: 'ja' },
  { name: '中文', code: 'zh-CN' },
];

export const HISTORICAL_PERSONAS: Persona[] = [
  {
    name: {
      ko: '세종대왕',
      en: 'King Sejong',
      ja: '世宗大王',
      'zh-CN': '世宗大王',
    },
    description: {
      ko: '조선의 가장 위대한 성군. 지혜롭고 자애로운 목소리로 대화합니다.',
      en: 'The greatest king of Joseon. Speaks with a wise and benevolent voice.',
      ja: '朝鮮最高の聖君。賢明で慈悲深い声で話します。',
      'zh-CN': '朝鲜最伟大的圣君。以智慧和仁慈的声音说话。',
    },
    avatarId: 'king-sejong',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'king-sejong')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '신사임당',
      en: 'Shin Saimdang',
      ja: '申師任堂',
      'zh-CN': '申师任堂',
    },
    description: {
      ko: '조선 중기의 문인이자 화가. 지혜롭고 온화한 어머니의 목소리로 이야기합니다.',
      en: 'A writer and painter of the mid-Joseon period. Speaks with a wise and gentle motherly voice.',
      ja: '朝鮮中期の文人であり画家。賢明で穏やかな母の声で話します。',
      'zh-CN': '朝鲜中期的文人和画家。用智慧和温和的母亲般的声音说话。',
    },
    avatarId: 'shin-saimdang',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'shin-saimdang')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '이순신',
      en: 'Yi Sun-sin',
      ja: '李舜臣',
      'zh-CN': '李舜臣',
    },
    description: {
      ko: '충무공 이순신 장군. 충성스럽고 강직한 어조로 이야기합니다.',
      en: 'Admiral Yi Sun-sin. Speaks with a loyal and resolute tone.',
      ja: '忠武公李舜臣将軍。忠実で剛直な口調で話します。',
      'zh-CN': '忠武公李舜臣将军。以忠诚和刚直的语气说话。',
    },
    avatarId: 'yi-sun-sin',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'yi-sun-sin')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '황진이',
      en: 'Hwang Jini',
      ja: '黄真伊',
      'zh-CN': '黄真伊',
    },
    description: {
      ko: '조선 최고의 기생이자 예술가. 우아하고 시적인 말투를 사용합니다.',
      en: 'The best gisaeng and artist of Joseon. Uses an elegant and poetic tone.',
      ja: '朝鮮最高の妓生であり芸術家。優雅で詩的な話し方をします。',
      'zh-CN': '朝鲜最优秀的妓生和艺术家。使用优雅而富有诗意的语调。',
    },
    avatarId: 'hwang-jini',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'hwang-jini')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '장금이',
      en: 'Jang-geum',
      ja: 'チャングム',
      'zh-CN': '长今',
    },
    description: {
      ko: '조선 왕실의 뛰어난 의녀이자 요리사. 상냥하고 지적인 목소리로 설명합니다.',
      en: 'An outstanding female physician and cook of the Joseon royal court. Explains with a kind and intelligent voice.',
      ja: '朝鮮王室の優れた医女であり料理人。優しく知的な声で説明します。',
      'zh-CN': '朝鲜王室杰出的医女和厨师。用亲切而充满智慧的声音解说。',
    },
    avatarId: 'jang-geum',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'jang-geum')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '초랭이',
      en: 'Choraengi',
      ja: 'チョレンイ',
      'zh-CN': '草랭이',
    },
    description: {
      ko: '재치와 유머가 넘치는 조선의 마당쇠. 재미있고 구수한 입담을 자랑합니다.',
      en: 'A witty and humorous servant from the Joseon Dynasty. Boasts entertaining and folksy storytelling.',
      ja: '機知とユーモアがあふれる朝鮮の庭師。面白く味のある語り口が自慢です。',
      'zh-CN': '充满机智和幽默的朝鲜仆人。以风趣和质朴的口才而自豪。',
    },
    avatarId: 'choraengi',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'choraengi')?.imageUrl ?? '',
  },
  {
    name: {
      ko: '역사 가이드',
      en: 'History Guide',
      ja: '歴史ガイド',
      'zh-CN': '历史向导',
    },
    description: {
      ko: '역사적 사실에 기반하여 정확한 정보를 전달하는 전문 가이드입니다.',
      en: 'A professional guide who delivers accurate information based on historical facts.',
      ja: '歴史的事実に基づいて正確な情報を伝える専門ガイドです。',
      'zh-CN': '根据历史事实提供准确信息的专业向导。',
    },
    avatarId: 'history-guide',
    avatarUrl:
      PlaceHolderImages.find((img) => img.id === 'history-guide')?.imageUrl ?? '',
  },
];
