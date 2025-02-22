import unittest
from scipy.io import wavfile
from Real_Time_Predict import kelime_say, hesapla_mfcc, tahmin_et
from Emotion_Analysis import analyze_sentiment, analyze_topic, translate_text

class TestKelimeSay(unittest.TestCase):
    def test_kelime_say(self):
        
        text1 = "Bu bir test cumlesidir."
        self.assertEqual(kelime_say(text1), 4, "Kelime sayisi hatali.")

        
        text2 = ""
        self.assertEqual(kelime_say(text2), 0, "Bos metin icin kelime sayisi 0 olmali.")

        
        text3 = "   Fazla   bosluklu   bir   cumle.   "
        self.assertEqual(kelime_say(text3), 4, "Fazla bosluklar dogru islenmedi.")

        
        text4 = "Bu, bir test: cumlesi; kelimeleri dogru sayar mi?"
        self.assertEqual(kelime_say(text4), 8, "Noktalama isaretleri yanlis degerlendirildi.")

class TestDuyguAnalizi(unittest.TestCase):
    def test_analyze_sentiment_positive(self):
        
        text = "Bu harika bir gun, cok mutluyum ve her sey mukemmel."
        polarity, subjectivity = analyze_sentiment(text)
        self.assertGreaterEqual(polarity, 0, "Pozitif cumle icin polarity degeri 0'dan buyuk olmali.")
        self.assertGreaterEqual(subjectivity, 0, "Subjectivity degeri 0 veya daha buyuk olmali.")
    
    def test_analyze_sentiment_negative(self):
        
        text = "Bu cok kotu bir gun, her sey berbat ve mutsuzum."
        polarity, subjectivity = analyze_sentiment(text)
        self.assertLessEqual(polarity, 0, "Negatif cumle icin polarity degeri 0'dan kucuk olmali.")
        self.assertGreaterEqual(subjectivity, 0, "Subjectivity degeri 0 veya daha buyuk olmali.")

    def test_analyze_sentiment_neutral(self):
        
        text = "Bu sadece siradan bir gun, hava durumu normal ve isler rutin devam ediyor."
        polarity, subjectivity = analyze_sentiment(text)
        self.assertAlmostEqual(polarity, 0, delta=0.2, msg="Notr cumle icin polarity degeri 0'a yakin olmali.")
        self.assertGreaterEqual(subjectivity, 0, "Subjectivity degeri 0 veya daha buyuk olmali.")

class TestKonuAnalizi(unittest.TestCase):
    def test_analyze_topic(self):
        
        # Metin 1
        text1 = "Yapay zeka ve makine ogrenimi hizla gelisiyor. Teknoloji, saglik, egitim ve daha pek cok alanda devrim yaratiyor."
        translated_text1 = translate_text(text1)  
        entities_info1 = analyze_topic(translated_text1)
        
        self.assertGreater(len(entities_info1), 0, "Konu analizi sonucu bos olmamalidir.")
        self.assertIn("Artificial intelligence", [entity["name"] for entity in entities_info1], "Yapay zeka konusu bulunamadi.")
        
        
        # Metin 2
        text2 = "Futbol takimlari yeni sezon icin hazirlik yapiyor. Dunyada futbol cok populer bir spor dalidir."
        translated_text2 = translate_text(text2)  
        entities_info2 = analyze_topic(translated_text2)
        
        self.assertGreater(len(entities_info2), 0, "Konu analizi sonucu bos olmamalidir.")
        self.assertIn("Football", [entity["name"] for entity in entities_info2], "Futbol konusu bulunamadi.")
        
        # Metin 3
        text3 = "Yeni kesifler bilim insanlarini heyecanlandiriyor. Evrenin sirlari hala tam olarak cozulemedi."
        translated_text3 = translate_text(text3)  
        entities_info3 = analyze_topic(translated_text3)
        
        self.assertGreater(len(entities_info3), 0, "Konu analizi sonucu bos olmamalidir.")
        self.assertIn("scientists", [entity["name"] for entity in entities_info3], "Bilim Adamlari konusu bulunamadi.")

if __name__ == '__main__':
    unittest.main()
