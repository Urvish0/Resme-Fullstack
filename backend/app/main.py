from .ui.gradio_app import demo

def main():
    print("\n" + "="*50)
    print("🚀 Launching ResMe 2.0")
    print("="*50)
    demo.launch(
        show_error=True,
        debug=True    
    )

if __name__ == "__main__":
    main()