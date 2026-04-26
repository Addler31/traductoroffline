import os
import asyncio
import flet as ft
from transformers import MarianMTModel, MarianTokenizer

# Global variable to store the main event loop
main_event_loop = None

def setup_marian_translator():
    """Initializes the tradutort offline with MarianMT."""
    try:
        print("Descargando/verifying modelos MarianMT...")

        # Modelos MarianMT para traducción ES-EN y EN-ES
        models = {
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-es": "Helsinki-NLP/opus-mt-en-es"
        }

        translators = {}
        tokenizers = {}

        for lang_pair, model_name in models.items():
            print(f"Cargando modelo {lang_pair}: {model_name}")

            # Cargar tokenizer y modelo
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            # Guardar en diccionario
            tokenizers[lang_pair] = tokenizer
            translators[lang_pair] = model
        print("✅ Todos los modelos MarianMT cargados correctamente.")
        print("Traductor offline listo para usar sin conexión a internet.")
        return translators, tokenizers
    except Exception as e:
        print(f"❌ Error inicializando MarianMT: {e}")
        return None, None

def main(page: ft.Page, translators=None, tokenizers=None):
    """Main function to launch the app."""
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()

    if translators is None or tokenizers is None:
        translators, tokenizers = setup_marian_translator()

        if not translators or not tokenizers:
            page.add(ft.Text("Error: No se pudo inicializar el traductor offline", color="red"))
            return

    page.title = "TraductOFF"
    page.window.width = 450
    page.window.height = 680
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.scroll = ft.ScrollMode.ADAPTIVE
    
    # --- UI COMPONENTS ---
    input_text = ft.TextField(
        label="Texto a traducir",
        multiline=True,
        min_lines=5,
        max_lines=12,
        hint_text="Escribe o pega aquí tu texto..."
    )
    output_text = ft.TextField(
        label="Resultado",
        multiline=True,
        read_only=True,
        min_lines=5,
        max_lines=12,
        bgcolor=ft.Colors.BLACK26
    )
    
    direction = ft.Dropdown(
        label="Idiomas",
        options=[
            ft.dropdown.Option("es-en", "Español a Inglés"),
            ft.dropdown.Option("en-es", "Inglés a Español"),
        ],
        value="es-en",
        width=300
    )
    
    loading_bar = ft.ProgressBar(width=400, color="blue", visible=False)
    
    # --- LÓGICA DE TRADUCCIÓN ---
    nonlocal_has_error = {"value": False}
    
    def translate_task(text: str, from_lang: str, to_lang: str,
                       output_widget: ft.TextField,
                       loading_bar: ft.ProgressBar,
                       btn_translate: ft.Button,
                       translators, tokenizers):
        """Translate the text."""
        try:
            if from_lang not in translators or to_lang not in translators:
                raise ValueError(f"Par de idiomas no soportado: {from_lang}-{to_lang}")

            model = translators[from_lang]
            tokenizer = tokenizers[to_lang]

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            def update_ui():
                output_widget.value = result
                nonlocal_has_error["value"] = False
                loading_bar.visible = False
                btn_translate.disabled = False
                page.update()

            main_event_loop.call_soon_threadsafe(update_ui)
        except Exception as e:
            update_ui_error =  f"Error: {type(e).__name__}: {e}"
            nonlocal_has_error["value"] = True
            loading_bar.visible = False
            btn_translate.disabled = False
            page.update()

    def on_translate_click(e):
        """Translate when the button is clicked."""
        text = input_text.value.strip()
        if not text:
            output_text.value = "Ingresa texto primero."
            page.update()
        return

    def on_clear_click(e):
        """Clear input text and output text."""
        input_text.value = ""
        output_text.value = ""
        return

    def on_thread_update(e):
      """Handles thread updates."""
      nonlocal_has_error = {"value": False}
      return

    def update_ui_error():
      """Handles error"""
      output_text.value = f"Error: {type(e).__name__}: {e}"
      nonlocal_has_error["value"] = True
      loading_bar.visible = False
      btn_translate.disabled = False
      page.update()

    def on_thread_safe(e):
      """Handles thread safe calls."""
      nonlocal_has_error = {"value": False}
      return
    
    def on_page_update(e):
        """Handles page updates"""
        nonlocal_has_error = {"value": False}
        return

    def app_main(page: ft.Page):
        return main(page, translators, tokenizers)
    
    page.add(build_layout())

if __name__ == "__main__":
    app_main(page, translators, tokenizers)
