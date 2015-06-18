require "nokogiri"
require "byebug"
require "utf8_utils"
require "cld"
 
BASE_DIR = "data/blogs_female" 
TARGET_DIR = "data/blogs_female_cleaned" 
 
Dir["#{BASE_DIR}/*"].each do |path|
  blog = Nokogiri::XML(File.open(path))
  posts = blog.xpath("//post")
 
  basename  = File.basename(path)
  new_path = File.join(TARGET_DIR, basename) 
   
  File.open(new_path, mode = "w") do |f|
    # remove delimiter newlines
    # #tidy_bytes replaces invalid utf-8 bytes with proper ones.
    posts.each do |post|
      begin
        clean_text = post.text.tidy_bytes.strip
        language = CLD.detect_language(clean_text)
        if language[:name] == "ENGLISH"
          f.print(clean_text)
        else
          puts "This is not English, but #{language[:name]}, right? #{clean_text}"
        end

      rescue ArgumentError, NoMethodError => e
        puts "I caught #{e}, I didn't include a post from #{new_path}, namely this one: #{post.text}." 
      end
    end
    
  end
  
end
