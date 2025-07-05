from new_functions import Functions as fnct

if __name__ == "__main__":
    f = fnct()
    while True:
        input1 = input("What do I do: ")
        match input1:
            case "discover":
                f.make_discoverable_and_listen()
            case "route":
                f.set_audio_output_to_aux()
            case "quit":
                break
        if "cmd:" in input1:
            cmd = input1.split(":")[1]
            f.send_media_command(cmd)
            exit()